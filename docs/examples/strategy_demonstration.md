---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Servicing Strategies

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)

In this example, we'll demonstrate the essential differences in scheduled servicing, unscheduled servicing, and tow-to-port repair strategies. Each of the examples demonstrated below will be based on the 2015 Dinwoodie, et al. paper, though the variations will be for demonstration purposes only.

A Jupyter notebook of this tutorial can be run from
`examples/strategy_demonstration.ipynb`locally, or through
[binder](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples).

## WOMBAT Setup and Variables

The vessels that will be changed in this demonstration are the field support vessel (FSV) with capability: "SCN", the heavy lift vessel (HLV) with capability: "LCN", and the tugboats, which have capability: "TOW".

```{note}
When running tow-to-port a `Port` configuration is also required, which will control the tugboats. However, the port costs will still be accounted for in the
`FixedCosts` class as port fees are assumed to be constant. These costs are not considered in this example, so differences in cost should be taken with a grain
of salt given the reduction of HLV and FSV operational and mobilization costs.
```

Scenario descriptions:
- **Scheduled**: exactly the same as the base case (fsv_scheduled.yaml and hlv_scheduled.yaml)
- **Unscheduled: requests**: the FSV and HLV are called to site when 10 requests that
  they can service are logged (fsv_requests.yaml and hlv_requests.yaml)
- **Unscheduled: downtime**: the FSV and HLV are called to site once the wind farm's
  operating level hits 90% or lower (fsv_downtime.yaml and hlv_downtime.yaml)
 - **Unscheduled: tow-to-port**: the FSV and HLV will be replaced with three identical
  tugboats (tugboat1.yaml, tugboat2.yaml, tugboat3.yaml), and all the failures associated
  with the FSV and HLV will be changed to capability "TOW" to trigger the tow-to-port
  repairs. These processes will be triggered on the first request (WOMBAT base
  assumption that can't be changed for now).

In this example, we will demonstrate how the results for the base case for the Dinwoodie, et al. example vary based on how each of the vessels are scheduled. The configuration details all remain the same, regardless of details, except for the strategy information, which is defined as follows:

This example is set up similarly to that of the validation cases to show how the results differ, and not a step-by-step guide for setting up the analyses. We refer the reader to the extensive [documentation](../API/index.md) and [How To example](how_to.md) for more information.

## Imports and notebook configuration

```{code-cell} ipython3
from copy import deepcopy
from time import perf_counter

import pandas as pd

from wombat.core import Simulation
from wombat.core.library import DINWOODIE

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.options.display.float_format = '{:,.2f}'.format
```

## Simulation and results setup

Here we're providing the names of the configuration files (found at: dinwoodie / config)
without their .yaml extensions (added in later) and the results that we want to compare
between simulations to understand some of the timing and cost trade-offs between
simulations.

The dictionary of keys and lists will be used to create the results data frame where the
keys will be the indices and the lists will be the row values for each of the above
configurations.

```{code-cell} ipython3
configs = [
    "base_scheduled",
    "base_requests",
    "base_downtime",
    "base_tow_to_port",
]

columns = deepcopy(configs) # Create a unique copy of the config names for column naming
results = {
    "availability - time based": [],
    "availability - production based": [],
    "capacity factor - net": [],
    "capacity factor - gross": [],
    "power production": [],
    "task completion rate": [],
    "annual direct O&M cost": [],
    "annual vessel cost": [],
    "ctv cost": [],
    "fsv cost": [],
    "hlv cost": [],
    "tow cost": [],
    "annual repair cost": [],
    "annual technician cost": [],
    "ctv utilization": [],
    "fsv utilization": [],
    "hlv utilization": [],
    "tow utilization": [],
}
```

## Run the simulations and display the results

```{code-cell} ipython3
timing_df = pd.DataFrame([], columns=["Load Time (min)", "Run Time (min)"], index=configs)
timing_df.index.name = "Scenario"

for config in configs:

    # Load the simulation
    start = perf_counter()
    sim = Simulation(DINWOODIE , f"{config}.yaml")
    end = perf_counter()
    timing_df.loc[config, "Load Time (min)"] = (end - start) / 60

    # Run the simulation
    start = perf_counter()
    sim.run()
    end = perf_counter()
    timing_df.loc[config, "Run Time (min)"] = (end - start) / 60

    # Gather the results of interest
    years = sim.metrics.events.year.unique().shape[0]
    mil = 1000000

    # Gather the high-level results for the simulation
    availability_time = sim.metrics.time_based_availability(frequency="project", by="windfarm")
    availability_production = sim.metrics.production_based_availability(frequency="project", by="windfarm")
    cf_net = sim.metrics.capacity_factor(which="net", frequency="project", by="windfarm")
    cf_gross = sim.metrics.capacity_factor(which="gross", frequency="project", by="windfarm")
    power_production = sim.metrics.power_production(frequency="project", by="windfarm")
    completion_rate = sim.metrics.task_completion_rate(which="both", frequency="project")
    parts = sim.metrics.events[["materials_cost"]].sum().sum()
    techs = sim.metrics.project_fixed_costs(frequency="project", resolution="low").operations[0]
    total = sim.metrics.events[["total_cost"]].sum().sum()

    # Gather the equipment costs and separate the results by equipment type
    equipment = sim.metrics.equipment_costs(frequency="project", by_equipment=True)
    equipment_sum = equipment.sum().sum()
    hlv = equipment[[el for el in equipment.columns if "Heavy Lift Vessel" in el]].sum().sum()
    fsv = equipment[[el for el in equipment.columns if "Field Support Vessel" in el]].sum().sum()
    ctv = equipment[[el for el in equipment.columns if "Crew Transfer Vessel" in el]].sum().sum()
    tow = equipment[[el for el in equipment.columns if "Tugboat" in el]].sum().sum()

    # Gather the equipment utilization data frame and separate the results by equipment type
    utilization = sim.metrics.service_equipment_utilization(frequency="project")
    hlv_ur = utilization[[el for el in utilization.columns if "Heavy Lift Vessel" in el]].mean().mean()
    fsv_ur = utilization[[el for el in utilization.columns if "Field Support Vessel" in el]].mean().mean()
    ctv_ur = utilization[[el for el in utilization.columns if "Crew Transfer Vessel" in el]].mean().mean()
    tow_ur = utilization[[el for el in utilization.columns if "Tugboat" in el]].mean().mean()

    # Log the results of interest
    results["availability - time based"].append(availability_time.values[0][0])
    results["availability - production based"].append(availability_production.values[0][0])
    results["capacity factor - net"].append(cf_net.values[0][0])
    results["capacity factor - gross"].append(cf_gross.values[0][0])
    results["power production"].append(power_production.values[0][0])
    results["task completion rate"].append(completion_rate.values[0][0])
    results["annual direct O&M cost"].append((total + techs) / mil / years)
    results["annual vessel cost"].append(equipment_sum / mil / years)
    results["ctv cost"].append(ctv / mil / years)
    results["fsv cost"].append(fsv / mil / years)
    results["hlv cost"].append(hlv / mil / years)
    results["tow cost"].append(tow / mil / years)
    results["annual repair cost"].append(parts / mil / years)
    results["annual technician cost"].append(techs / mil / years)
    results["ctv utilization"].append(ctv_ur)
    results["fsv utilization"].append(fsv_ur)
    results["hlv utilization"].append(hlv_ur)
    results["tow utilization"].append(tow_ur)

    # Clear the logs
    sim.env.cleanup_log_files()

timing_df
```

```{code-cell} ipython3
results_df = pd.DataFrame(results.values(), columns=columns, index=results.keys()).fillna(0)
results_df
```
