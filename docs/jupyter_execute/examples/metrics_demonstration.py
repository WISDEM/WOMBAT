#!/usr/bin/env python
# coding: utf-8

# # Demonstration of the Metrics To-Date
#
# For a complete list of metrics and their documentation, please see the API Metrics [documentation](../API/simulation_api.md#metrics-computation).
#
# This demonstration will rely on the results produced in the "How To" notebook and serves as an extension of the API documentation to show what the results will look like depending on what inputs are provided.

# In[1]:


from pprint import pprint

import pandas as pd

from wombat.core import Metrics, Simulation


# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


# ## Table of Contents
#
# Below is a list of top-level sections to demonstrate how to use WOMBAT's `Metrics` class methods and an explanation of each individual metric.
#  - [Setup](#setup): Running a simulation to gather the results
#  - [Availability](#availability): Time-based and energy-based availability
#  - [Capacity Factor](#capacity-factor): Gross and net capacity factor
#  - [Task Completion Rate](#task-completion-rate): Task completion metrics
#  - [Equipment Costs](#equipment-costs): Cost breakdowns by servicing equipment
#  - [Service Equipment Utilization Rate](#service-equipment-utilization-rate): Utilization of servicing equipment
#  - [Vessel-Crew Hours at Sea](#vessel-crew-hours-at-sea): Number of crew or vessel hours spent at sea
#  - [Number of Tows](#number-of-tows): Number of tows breakdowns
#  - [Labor Costs](#labor-costs): Breakdown of labor costs
#  - [Equipment and Labor Costs](#equipment-and-labor-costs): Combined servicing equipment and labor cost breakdown
#  - [Component Costs](#component-costs): Materials costs
#  - [Fixed Cost Impacts](#fixed-cost-impacts): Total fixed costs
#  - [OpEx](#opex): Project OpEx
#  - [Process Times](#process-times): Timing of various stages of repair and maintenance
#  - [Power Production](#power-production): Potential and actually produced power
#  - [Net Present Value](#net-present-value): Project NPV calculator
#  - [PySAM-Powered Results](#pysam-powered-results): PySAM results, if configuration is provided

# <a id="setup"></a>
# ## Setup
#
# The simulations from the How To notebook are going to be rerun as it is not recommended to create a Metrics class from scratch due to the
# large number of inputs that are required and the initialization is provided in the simulation API's run method.
#
# To simplify this process, a feature has been added to save the simulation outputs required to generate the Metrics inputs and a method to reload those outputs as inputs.

# In[2]:


sim = Simulation("DINWOODIE", "base.yaml")

# Both of these parameters are True by default for convenience
sim.run(create_metrics=True, save_metrics_inputs=True)

# Load the metrics data
fpath = sim.env.metrics_input_fname.parent
fname = sim.env.metrics_input_fname.name
metrics = Metrics.from_simulation_outputs(fpath, fname)

# Delete the log files now that they're loaded in
sim.env.cleanup_log_files(log_only=False)

# Alternatively, in this case because the simulation was run, we can use the
# following for convenience convenience only
metrics = sim.metrics


# <a id="availability"></a>
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


# <a id="capacity-factor"></a>
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


# <a id="task-completion-rate"></a>
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


# <a id="equipment-costs"></a>
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
print(f"Project total: ${total.values[0][0] / metrics.project_capacity:,.2f}/MW")


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


# <a id="service-equipment-utilization-rate"></a>
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


# <a id="vessel-crew-hours-at-sea"></a>
# ## Vessel-Crew Hours at Sea
#
# The number of vessel hours or crew hours at sea for offshore wind power plant simulations.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_equipment` options:
#  - `True`: computed for each vessel simulated
#  - `False`: computed only as a sum of all vessels
#
# `vessel_crew_assumption`: A dictionary of vessel names (`ServiceEquipment.settings.name`, but also found at `Metrics.service_equipment_names`) and the number of crew onboard at any given time. The application of this assumption transforms the results from vessel hours at sea to crew hours at sea.

# In[24]:


# Project total, not broken out by vessel
metrics.vessel_crew_hours_at_sea(frequency="project", by_equipment=False)


# In[25]:


# Annual project totals, broken out by vessel
metrics.vessel_crew_hours_at_sea(frequency="annual", by_equipment=True)


# <a id="number-of-tows"></a>
# ## Number of Tows
#
# The number of tows metric will only produce results if any towing actually occurred, otherwise a dataframe with a value of 0 is returned, as is the case in this demonstration.
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `by_tug` options:
#  - `True`: computed for each tugboat (towing vessel)
#  - `False`: computed only as a sum of all tugboats
#
# `by_direction` options:
#  - `True`: computed for each direction (to port or to site)
#  - `False`: computed as a sum for both directions

# In[26]:


# Project Total
# NOTE: This example has no towing, so it will return 0
metrics.number_of_tows(frequency="project")


# <a id="labor-costs"></a>
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

# In[27]:


# Project total at the whole windfarm level
total = metrics.labor_costs(frequency="project", by_type=False)
print(f"Project total: ${total.values[0][0] / metrics.project_capacity:,.2f}/MW")


# In[28]:


# Project totals for each type of labor
# NOTE: Only salaried labor was defined for thesese analyses
metrics.labor_costs(frequency="project", by_type=True)


# In[29]:


# Project annual totals for all labor
metrics.labor_costs(frequency="annual", by_type=False)


# In[30]:


# Project monthly totals for all labor
metrics.labor_costs(frequency="monthly", by_type=False)


# In[31]:


# Project month-by-year totals for all labor
# NOTE: This is limited to the first two years only
metrics.labor_costs(frequency="month-year", by_type=False).head(24)


# <a id="equipment-and-labor-costs"></a>
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
# **NOTE:** For this breakdown the expense category (reason) is distributed across the rows in addition to time.
#
#
# `reason` definitions:
#  - Maintenance: routine maintenance
#  - Repair: unscheduled maintenance, ranging from inspections to replacements
#  - Weather Delay: Any delays caused by unsafe weather conditions
#  - No Requests: Equipment and labor is active, but there are no repairs or maintenance tasks to be completed
#  - Not in Shift: Any time outside of the operating hours of the windfarm

# In[32]:


# Project totals
metrics.equipment_labor_cost_breakdowns(frequency="project", by_category=False)


# In[33]:


# Project totals by each category
metrics.equipment_labor_cost_breakdowns(frequency="project", by_category=True)


# In[34]:


# Project annual totals
# NOTE: This is limited to the first two years
metrics.equipment_labor_cost_breakdowns(frequency="annual", by_category=False).head(10)


# In[35]:


# Project monthly totals
# NOTE: This is limited to the first two years
metrics.equipment_labor_cost_breakdowns(frequency="monthly", by_category=False).head(10)


# In[36]:


# Project month-by-year totals
# NOTE: This is limited to the first two years
metrics.equipment_labor_cost_breakdowns(frequency="month-year", by_category=False).head(
    20
)


# <a id="component-costs"></a>
# ## Component Costs
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
# **NOTE:** For this breakdown the expense category (reason) is distributed across the rows in addition to time.
#
#
# `action` definitions:
#  - maintenance: routine maintenance
#  - repair: unscheduled maintenance, ranging from inspections to replacements
#  - delay: Any delays caused by unsafe weather conditions or not being able to finish a process within a single shift

# In[37]:


# Project totals by component
metrics.component_costs(frequency="project", by_category=False, by_action=False)


# In[38]:


# Project totals by each category and action type
metrics.component_costs(frequency="project", by_category=True, by_action=True)


# In[39]:


# Project annual totals by category
# NOTE: This is limited to the first two years
metrics.component_costs(frequency="annual", by_category=True, by_action=False).head(28)


# In[40]:


# Project monthly totals
# NOTE: This is limited to the first two months
metrics.component_costs(frequency="monthly", by_category=True, by_action=False).head(28)


# In[41]:


# Project month-by-year totals
# NOTE: This is limited to the first two months
metrics.component_costs(frequency="month-year", by_category=True, by_action=False).head(
    28
)


# <a id="fixed-cost-impacts"></a>
# ## Fixed Cost Impacts
#
# Here, we will go through the various input definitions to get the fixed cost data
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `resolution` options:
#  - high: computed across the lowest itemized cost levels
#  - medium: computed across overarching cost levels
#  - low: computed as single total

# In[42]:


# The resolution hierarchy for fixed costs
pprint(metrics.fixed_costs.hierarchy)


# In[43]:


# Project totals at the highest level
metrics.project_fixed_costs(frequency="project", resolution="low")


# In[44]:


# Project totals at the medium level
metrics.project_fixed_costs(frequency="project", resolution="medium")


# In[45]:


# Project totals at the lowest level
metrics.project_fixed_costs(frequency="project", resolution="high")


# In[46]:


# Project annualized totals at the medium level
metrics.project_fixed_costs(frequency="annual", resolution="medium")


# In[47]:


# Project annualized totals at the low level by month
metrics.project_fixed_costs(frequency="annual", resolution="low")


# In[48]:


# Project annualized totals at the low level by year and month
metrics.project_fixed_costs(frequency="month-year", resolution="low").head(28)


# <a id="opex"></a>
# ## OpEx
#
# Here, we will go through the various input definitions to get the project OpEx
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis

# In[49]:


metrics.opex("project")


# In[50]:


metrics.opex("annual")


# In[51]:


metrics.opex("monthly")


# In[52]:


metrics.opex("month-year").head(28)


# In[53]:


port_fees = pd.DataFrame(
    [],
    columns=["port_fees"],
    index=metrics.labor_costs(frequency="month-year", by_type=False).index,
)
port_fees.fillna(0)


# <a id="process-times"></a>
# ## Process Times
#
# There are no inputs for the process timing as it is a slow calculation, so aggregation is left to the user for now. The results corresond to the number of hours required to complete any of the repair or maintenance activities.

# In[54]:


# Project totals at the project level
total = metrics.process_times()
total


# <a id="power-production"></a>
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

# In[55]:


# Project total at the whole windfarm level
total = metrics.power_production(frequency="project", by_turbine=False)
total


# In[56]:


# Project totals at the turbine level
metrics.power_production(frequency="project", by_turbine=True)


# In[57]:


# Project annual totals for the windfarm
metrics.power_production(frequency="annual", by_turbine=False)


# In[58]:


# Project monthly totals for the windfarm
metrics.power_production(frequency="monthly", by_turbine=False)


# In[59]:


# Project month-by-year totals for the windfarm
# NOTE: This is limited to the first two years only
metrics.power_production(frequency="month-year", by_turbine=False).head(24)


# <a id="net-present-value"></a>
# ## Net Present Value
#
# Here, we will go through the various input definitions to get the project NPV
#
# `frequency` options:
#  - project: computed across the whole simulation
#  - annual: computed on a yearly basis
#  - monthly: computed across years on a monthly basis
#  - month-year: computed on a month-by-year basis
#
# `discount_rate`: The rate of return that could be earned on alternative investments, by default 0.025.
#
# `offtake_price`: Price of energy, per MWh, by default 80.

# In[60]:


metrics.npv("project")


# In[61]:


metrics.opex("annual")


# In[62]:


metrics.opex("monthly")


# In[63]:


metrics.opex("month-year").head(28)


# <a id="pysam-powered-results"></a>
# ## PySAM-Powered Results
#
# For a number of project financial metrics, the PySAM library is utilized.
#
# <div class="alert alert-block alert-warning">
# <b>NOTE:</b> If a "SAM_settings" file is not provided to the simulation, then the following metrics will not be able to be calculated and will raise a `NotImplementedError`.
# </div>
#
# With the above warning in mind, the appropriate simulation outputs are provided as inputs to PySAM upon initialization to ensure all values are aligned.
#
# ### Net Present Value (NPV)

# In[64]:


try:
    npv = metrics.pysam_npv()
    print(f"NPV: ${npv:,.0f}")
except NotImplementedError as e:
    print(e)


# ### Real Levelized Cost of Energy (LCOE)

# In[65]:


try:
    lcoe = metrics.pysam_lcoe_real()
    print(f"Real LCOE: ${lcoe:,.2f}/kW")
except NotImplementedError as e:
    print(e)


# ### Nominal Levelized Cost of Energy (LCOE)

# In[66]:


try:
    lcoe = metrics.pysam_lcoe_nominal()
    print(f"Nominal LCOE: ${lcoe:,.2f}/kW")
except NotImplementedError as e:
    print(e)


# ### After-tax Internal Return Rate (IRR)

# In[67]:


try:
    npv = metrics.pysam_irr()
    print(f"IRR: {npv:,.1f}%")
except NotImplementedError as e:
    print(e)


# ### One Data Frame to Rule Them All
#
# For this demonstration we will manually load a PySAM settings file and trigger the setup for demonstration purposes, but it should be noted that this practice should be avoided.

# In[68]:


try:
    metrics.pysam_all_outputs()
except NotImplementedError as e:
    print(e)


# In[ ]:
