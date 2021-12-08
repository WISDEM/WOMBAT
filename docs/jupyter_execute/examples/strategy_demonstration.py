#!/usr/bin/env python
# coding: utf-8

# # Servicing Equipment Strategy Demonstration
#
# ### National Renewable Energy Laboratory
#
# #### Rob Hammond
#
# ##### 7 December 2021
#
# In this example, we will demonstrate how the results for the base case for the Dinwoodie, et al. example vary based on how each of the vessels are scheduled. The configuration details all remain the same, regardless of details, except for the strategy information, which is defined as follows:
#  - **strategy_scheduled**: exactly the same as the base case (fsv_scheduled.yaml and hlv_scheduled.yaml)
#  - **strategy_requests**: the FSV and HLV are called to site when 10 requests that they can service are logged (fsv_requests.yaml and hlv_requests.yaml)
#  - **strategy_downtime**: the FSV and HLV are called to site once the windfarm's operating level hits 90% or lower (fsv_downtime.yaml and hlv_downtime.yaml)
#
#  This example is set up similar to that of the validation cases to show how the results differ, and not a step-by-step guide for setting up the analyses. We refer the reader to the exensive [documentation](../API/index.md) and [How To example](how_to.md) for more on the specifics.

# In[1]:


import pandas as pd
from copy import deepcopy
from pprint import pprint
from time import perf_counter

from wombat.core import Simulation
from wombat.core.library import DINWOODIE


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.options.display.float_format = "{:,.2f}".format


# In[2]:


configs = [
    "base_scheduled",
    "base_requests",
    "base_downtime",
]
columns = deepcopy(configs)
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
    "annual repair cost": [],
    "annual technician cost": [],
    "ctv utilization": [],
    "fsv utilization": [],
    "hlv utilization": [],
}


# In[3]:


print(f"{'name'.rjust(24)} | {'loading'} | running")
for config in configs:
    print(f"{config.rjust(24)}", end=" | ")

    # Load the simulation
    start = perf_counter()
    sim = Simulation.from_config(DINWOODIE / "config" / f"{config}.yaml")
    end = perf_counter()
    print(f"{(end - start) / 60:5.2f} m", end=" | ")

    # Run the simulation
    start = perf_counter()
    sim.run()
    end = perf_counter()
    print(f"{(end - start) / 60:2.2f} m")

    # Gather the results of interest
    years = sim.metrics.events.year.unique().shape[0]
    mil = 1000000

    availability_time = sim.metrics.time_based_availability(
        frequency="project", by="windfarm"
    )
    availability_production = sim.metrics.production_based_availability(
        frequency="project", by="windfarm"
    )
    cf_net = sim.metrics.capacity_factor(
        which="net", frequency="project", by="windfarm"
    )
    cf_gross = sim.metrics.capacity_factor(
        which="gross", frequency="project", by="windfarm"
    )
    power_production = sim.metrics.power_production(
        frequency="project", by_turbine=False
    ).values[0][0]
    completion_rate = sim.metrics.task_completion_rate(
        which="both", frequency="project"
    )
    parts = sim.metrics.events[["materials_cost"]].sum().sum()
    techs = sim.metrics.project_fixed_costs(
        frequency="project", resolution="low"
    ).operations[0]
    total = sim.metrics.events[["total_cost"]].sum().sum()

    equipment = sim.metrics.equipment_costs(frequency="project", by_equipment=True)
    equipment_sum = equipment.sum().sum()
    hlv = (
        equipment[[el for el in equipment.columns if "Heavy Lift Vessel" in el]]
        .sum()
        .sum()
    )
    fsv = (
        equipment[[el for el in equipment.columns if "Field Support Vessel" in el]]
        .sum()
        .sum()
    )
    ctv = (
        equipment[[el for el in equipment.columns if "Crew Transfer Vessel" in el]]
        .sum()
        .sum()
    )

    utilization = sim.metrics.service_equipment_utilization(frequency="project")
    hlv_ur = (
        utilization[[el for el in utilization.columns if "Heavy Lift Vessel" in el]]
        .mean()
        .mean()
    )
    fsv_ur = (
        utilization[[el for el in utilization.columns if "Field Support Vessel" in el]]
        .mean()
        .mean()
    )
    ctv_ur = (
        utilization[[el for el in utilization.columns if "Crew Transfer Vessel" in el]]
        .mean()
        .mean()
    )

    # Log the results of interest
    results["availability - time based"].append(availability_time)
    results["availability - production based"].append(availability_production)
    results["capacity factor - net"].append(cf_net)
    results["capacity factor - gross"].append(cf_gross)
    results["power production"].append(power_production)
    results["task completion rate"].append(completion_rate)
    results["annual direct O&M cost"].append((total + techs) / mil / years)
    results["annual vessel cost"].append(equipment_sum / mil / years)
    results["ctv cost"].append(ctv / mil / years)
    results["fsv cost"].append(fsv / mil / years)
    results["hlv cost"].append(hlv / mil / years)
    results["annual repair cost"].append(parts / mil / years)
    results["annual technician cost"].append(techs / mil / years)
    results["ctv utilization"].append(ctv_ur)
    results["fsv utilization"].append(fsv_ur)
    results["hlv utilization"].append(hlv_ur)

    # Clear the logs
    sim.env.cleanup_log_files(log_only=False)


# In[4]:


results_df = pd.DataFrame(
    results.values(), columns=columns, index=results.keys()
).fillna(0)
results_df


# In[ ]:
