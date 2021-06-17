#!/usr/bin/env python
# coding: utf-8

# # Demonstration of the Metrics To-Date
#
# For a complete list of metrics and their documentation, please see the API Metrics [documentation](../API/simulation_api.md#metrics-computation).
#
# This demonstration will rely on the results produced in the "How To" notebook.

# In[1]:


from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wombat.core import Metrics, Simulation


get_ipython().run_line_magic("matplotlib", "inline")
pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


# ## Setup
#
# The simulations from the How To notebook are going to be rerun as it is not recommended to create a Metrics class from scratch due to the
# large number of inputs that are required and the initialization is provided in the simulation API's run method.

# In[2]:


simulation_name = "example_dinwoodie"

sim = Simulation(simulation_name, "DINWOODIE", "base.yaml")
sim.run()

# For convenience only
metrics = sim.metrics


# ## Availability
#
# There are two methods to produce availability, which have their own function calls:
#  - energy: `production_based_availability`
#  - time: `time_based_availability`
#
# Here, we will go through the various input definitions to get time-based availability data as both methods use the same inputs, and provide outputs in the same format.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by` options:
#  - windfarm: computed across all turbines
#  - turbine: computed for each turbine

# In[3]:


# Project total at the whole windfarm level
total = metrics.time_based_availability(frequency="project", by="windfarm")
print(f"Project total: {total * 100:.1f}%")


# In[4]:


# Project total at the turbine level
metrics.time_based_availability(frequency="project", by="turbine")


# In[5]:


# Project annual totals at the windfarm level
metrics.time_based_availability(frequency="annual", by="windfarm")


# In[6]:


# Project monthly totals at the windfarm level
metrics.time_based_availability(frequency="monthly", by="windfarm")


# In[7]:


# Project month-by-year totals at the windfarm level
# NOTE: This is limited to the first two years for cleanliness of the notebook
metrics.time_based_availability(frequency="month-year", by="windfarm").head(24)


# ## Capacity Factor
#
# Here, we will go through the various input definitions to get capacity factor data. The inputs are very similar to that of the availability calculation.
#
# `which` options:
#  - net: net capcity factor, actual production
#  - gross: gross capacity factor, potential production
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by` options:
#  - windfarm: computed across all turbines
#  - turbine: computed for each turbine

# In[8]:


# Project total at the whole windfarm level
cf = metrics.capacity_factor(which="net", frequency="project", by="windfarm")
print(f"  Net Capacity Factor: {cf:.2f}%")

cf = metrics.capacity_factor(which="gross", frequency="project", by="windfarm")
print(f"Gross Capacity Factor: {cf:.2f}%")


# In[9]:


# Project total at the turbine level
metrics.capacity_factor(which="net", frequency="project", by="turbine")


# In[10]:


# Project annual totals at the windfarm level
metrics.capacity_factor(which="net", frequency="annual", by="windfarm")


# In[11]:


# Project monthly totals at the windfarm level
metrics.capacity_factor(which="net", frequency="monthly", by="windfarm")


# In[12]:


# Project month-by-year totals at the windfarm level
# NOTE: This is limited to the first two years for cleanliness of the notebook
metrics.capacity_factor(which="net", frequency="month-year", by="windfarm").head(24)


# ## Task Completion Rate
#
# Here, we will go through the various input definitions to get the task completion rates. The inputs are very similar to that of the availability calculation.
#
# `which` options:
#  - scheduled: scheduled maintenance only (classified as maintenace tasks in inputs)
#  - unscheduled: unscheduled maintenance only (classified as failure events in inputs)
#  - both:
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis

# In[13]:


# Project total at the whole windfarm level
total = metrics.task_completion_rate(which="scheduled", frequency="project")
print(f"  Scheduled Task Completion Rate: {total * 100:.0f}%")

total = metrics.task_completion_rate(which="unscheduled", frequency="project")
print(f"Unscheduled Task Completion Rate: {total * 100:.0f}%")

total = metrics.task_completion_rate(which="both", frequency="project")
print(f"    Overall Task Completion Rate: {total * 100:.0f}%")


# In[14]:


# Project annual totals at the windfarm level
metrics.task_completion_rate(which="both", frequency="annual")


# In[15]:


# Project monthly totals at the windfarm level
metrics.task_completion_rate(which="both", frequency="monthly")


# In[16]:


# Project month-by-year totals at the windfarm level
# NOTE: This is limited to the first two years for cleanliness of the notebook
metrics.task_completion_rate(which="both", frequency="month-year").head(24)


# ## Equipment Costs
#
# Here, we will go through the various input definitions to get the equipment cost data.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_equipment` options:
#  - `True`: computed across all equipment used
#  - `False`: computed for each piece of equipment

# In[17]:


# Project total at the whole windfarm level
total = metrics.equipment_costs(frequency="project", by_equipment=False)
print(f"Project total: ${total / metrics.project_capacity:,.2f}/MW")


# In[18]:


# Project totals at the equipment level
metrics.equipment_costs(frequency="project", by_equipment=True)


# In[19]:


# Project annual totals at the windfarm level
metrics.equipment_costs(frequency="annual", by_equipment=False)


# In[20]:


# Project monthly totals at the equipment level
metrics.equipment_costs(frequency="monthly", by_equipment=True)


# In[21]:


# Project month-by-year totals at the equipment level
# NOTE: This is limited to the two years only
metrics.equipment_costs(frequency="month-year", by_equipment=True).head(24)


# ## Service Equipment Utilization Rate
#
# Here, we will go through the various input definitions to get the service equipment utiliztion rates.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis

# In[22]:


# Project totals at the project level
total = metrics.service_equipment_utilization(frequency="project")
total


# In[23]:


# Annualized project totals
total = metrics.service_equipment_utilization(frequency="annual")
total


# ## Labor Costs
#
# Here, we will go through the various input definitions to get the labor cost data.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_type` options:
#  - `True`: computed across each labor type
#  - `False`: computed for both labor types used

# In[24]:


# Project total at the whole windfarm level
total = metrics.labor_costs(frequency="project", by_type=False)
print(f"Project total: ${total / metrics.project_capacity:,.2f}/MW")


# In[25]:


# Project totals for each type of labor
# NOTE: Only salaried labor was defined for thesese analyses
metrics.labor_costs(frequency="project", by_type=True)


# In[26]:


# Project annual totals for all labor
metrics.labor_costs(frequency="annual", by_type=False)


# In[27]:


# Project monthly totals for all labor
metrics.labor_costs(frequency="monthly", by_type=False)


# In[28]:


# Project month-by-year totals for all labor
# NOTE: This is limited to the first two years only
metrics.labor_costs(frequency="month-year", by_type=False).head(24)


# ## Equipment and Labor Costs
#
# Here, we will go through the various input definitions to get the equipment and labor cost data broken out by expense categories.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_category` options:
#  - `True`: computed across as equipment, plus each labor type, plus totals
#  - `False`: computed as single total
#
# ##### **NOTE:** For this breakdown the expense category (reason) is distributed across the rows in addition to time.
#
#
# `reason` definitions:
#  - Maintenance: routine maintenance
#  - Repair: unscheduled maintenance, ranging from inspections to replacements
#  - Weather Delay: Any delays caused by unsafe weather conditions
#  - No Requests: Equipment and labor is active, but there are no repairs or maintenance tasks to be completed
#  - Not in Shift: Any time outside of the operating hours of the windfarm

# In[29]:


# Project totals
metrics.equipment_labor_cost_breakdowns(frequency="project", by_category=False)


# In[30]:


# Project totals by each category
metrics.equipment_labor_cost_breakdowns(frequency="project", by_category=True)


# In[31]:


# Project annual totals
# NOTE: This is limited to the first two years
metrics.equipment_labor_cost_breakdowns(frequency="annual", by_category=False).head(10)


# In[32]:


# Project monthly totals
# NOTE: This is limited to the first two years
metrics.equipment_labor_cost_breakdowns(frequency="monthly", by_category=False).head(10)


# In[33]:


# Project month-by-year totals
# NOTE: This is limited to the first two years
metrics.equipment_labor_cost_breakdowns(frequency="month-year", by_category=False).head(
    20
)


# ## Component
#
# Here, we will go through the various input definitions to get the component cost data broken out by various categories.
#
# **NOTE**: It should be noted that the the component costs will not sum up to the whole project operations costs because of delays that are not associated with any repair or maintenance task, such as no requests needing to be processed.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_category` options:
#  - `True`: computed across each cost category (includes total)
#  - `False`: computed as single total
#
#  `by_action` options:
#  - `True`: computed by each of "repair", "maintenance", and "delay"
#  - `False`: computed as single total
#
# ##### **NOTE:** For this breakdown the expense category (reason) is distributed across the rows in addition to time.
#
#
# `action` definitions:
#  - maintenance: routine maintenance
#  - repair: unscheduled maintenance, ranging from inspections to replacements
#  - delay: Any delays caused by unsafe weather conditions or not being able to finish a process within a single shift

# In[34]:


# Project totals by component
metrics.component_costs(frequency="project", by_category=False, by_action=False)


# In[35]:


# Project totals by each category and action type
metrics.component_costs(frequency="project", by_category=True, by_action=True)


# In[36]:


# Project annual totals by category
# NOTE: This is limited to the first two years
metrics.component_costs(frequency="annual", by_category=True, by_action=False).head(28)


# In[37]:


# Project monthly totals
# NOTE: This is limited to the first two months
metrics.component_costs(frequency="monthly", by_category=True, by_action=False).head(28)


# In[38]:


# Project month-by-year totals
# NOTE: This is limited to the first two months
metrics.component_costs(frequency="month-year", by_category=True, by_action=False).head(
    28
)


# ## Fixed Cost Impacts
#
# Here, we will go through the various input definitions to get the fixed cost data
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#
# `resolution` options:
#  - high: computed across the lowest itemized cost levels
#  - medium: computed across overarching cost levels
#  - low: computed as single total

# In[39]:


# The resolution hierarchy for fixed costs
pprint(metrics.fixed_costs.hierarchy)


# In[40]:


# Project totals at the highest level
metrics.project_fixed_costs(frequency="project", resolution="low")


# In[41]:


# Project totals at the medium level
metrics.project_fixed_costs(frequency="project", resolution="medium")


# In[42]:


# Project totals at the lowest level
metrics.project_fixed_costs(frequency="project", resolution="high")


# In[43]:


# Project annualized totals at the medium level
metrics.project_fixed_costs(frequency="annual", resolution="medium")


# ## Process Times
#
# There are no inputs for the process timing as it is a slow calculation, so aggregation is left to the user for now. The results corresond to the number of hours required to complete any of the repair or maintenance activities.

# In[44]:


# Project totals at the project level
total = metrics.process_times()
total


# ## Power Production
#
# Here, we will go through the various input definitions to get the power production data.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_turbine` options:
#  - `True`: computed for each turbines
#  - `False`: computed for the whole windfarm

# In[45]:


# Project total at the whole windfarm level
total = metrics.power_production(frequency="project", by_turbine=False)
total


# In[46]:


# Project totals at the turbine level
metrics.power_production(frequency="project", by_turbine=True)


# In[47]:


# Project annual totals for the windfarm
metrics.power_production(frequency="annual", by_turbine=False)


# In[48]:


# Project monthly totals for the windfarm
metrics.power_production(frequency="monthly", by_turbine=False)


# In[49]:


# Project month-by-year totals for the windfarm
# NOTE: This is limited to the first two years only
metrics.power_production(frequency="month-year", by_turbine=False).head(24)


# ## PySAM-Powered Results
#
# For a number of project financial metrics, the PySAM library is utilized.
#
# <div class="alert alert-block alert-warning">
# <b>NOTE:</b> A PySAM file needs to be defined in your library's windfarm directory. This part of the analysis is highly dependent on the inputs provided, and there is not a validated input data sheet used, this is merely a functionality demonstration.
# </div>
#
# With the above warning in mind, the appropriate simulation outputs are provided as inputs to PySAM upon initialization to ensure all values are aligned.
#
# ### Net Present Value (NPV)

# In[50]:


npv = metrics.pysam_npv()
print(f"NPV: ${npv:,.0f}")


# ### Real Levelized Cost of Energy (LCOE)

# In[51]:


lcoe = metrics.pysam_lcoe_real()
print(f"Real LCOE: ${lcoe:,.2f}/kW")


# ### Nominal Levelized Cost of Energy (LCOE)

# In[52]:


lcoe = metrics.pysam_lcoe_nominal()
print(f"Nominal LCOE: ${lcoe:,.2f}/kW")


# ### After-tax Internal Return Rate (IRR)

# In[53]:


npv = metrics.pysam_irr()
print(f"IRR: {npv:,.1f}%")


# ### One Data Frame to Rule Them All

# In[54]:


metrics.pysam_all_outputs()


# In[55]:


sim.env.cleanup_log_files(log_only=False)


# In[ ]:
