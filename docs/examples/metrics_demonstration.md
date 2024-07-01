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

(metrics-demo)=
# Demonstration of the Available Metrics

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)

For a complete list of metrics and their documentation, please see the API Metrics
[documentation](simulation-api:metrics).

This demonstration will rely on the results produced in the "How To" notebook and serves
as an extension of the API documentation to show what the results will look like
depending on what inputs are provided.

A Jupyter notebook of this tutorial can be run from
`examples/metrics_demonstration.ipynb` locally, or through
[binder](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples).

```{code-cell} ipython3
:tags: ["output_scroll"]
from pprint import pprint
from functools import partial

import pandas as pd
from pandas.io.formats.style import Styler

from wombat.core import Simulation, Metrics
from wombat.utilities import plot

# Clean up the aesthetics for the pandas outputs
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 10)
style = partial(
  Styler,
  table_attributes='style="font-size: 14px; grid-column-count: 6"',
  precision=2,
  thousands=",",
)
```

(metrics-demo:toc)=
## Table of Contents

Below is a list of top-level sections to demonstrate how to use WOMBAT's `Metrics`
class methods and an explanation of each individual metric.

If you don't see a metric or result computation that is core to your work, please submit
an [issue](https://github.com/WISDEM/WOMBAT/issues/new) with details on what the metric
is, and how it should be computed.

- [Setup](metrics-demo:setup): Running a simulation to gather the results
- [Common Parameters](metrics-demo:common-parameters): Explanation of frequently used
  parameter settings
- [Availability](metrics-demo:availability): Time-based and energy-based availability
- [Capacity Factor](metrics-demo:cf): Gross and net capacity factor
- [Task Completion Rate](metrics-demo:task-completion): Task completion metrics
- [Equipment Costs](metrics-demo:equipment-costs): Cost breakdowns by servicing equipment
- [Service Equipment Utilization Rate](metrics-demo:utilization-rate): Utilization
  of servicing equipment
- [Vessel-Crew Hours at Sea](metrics-demo:vessel-crew-hours): Number of crew or vessel hours
  spent at sea
- [Number of Tows](metrics-demo:n-tows): Number of tows breakdowns
- [Labor Costs](metrics-demo:labor-costs): Breakdown of labor costs
- [Equipment and Labor Costs](metrics-demo:equipment-labor-costs): Combined servicing equipment
  and labor cost breakdown
- [Emissions](metrics-demo:emissions): Emissions of servicing equipment based on activity
- [Component Costs](metrics-demo:component-costs): Materials costs
- [Fixed Cost Impacts](metrics-demo:fixed-costs): Total fixed costs
- [OpEx](metrics-demo:opex): Project OpEx
- [Process Times](metrics-demo:process-times): Timing of various stages of repair and maintenance
- [Power Production](metrics-demo:power-production): Potential and actually produced power
- [Net Present Value](metrics-demo:npv): Project NPV calculator

(metrics-demo:setup)=
## Setup

The simulations from the *How To* notebook are going to be rerun as it is not
recommended to create a Metrics class from scratch due to the large number of inputs
that are required, and the initialization is provided in the simulation API's run method.

To simplify this process, a feature has been added to save the simulation outputs
required to generate the Metrics inputs and a method to reload those outputs as inputs.

```{code-cell} ipython3
:tags: ["output_scroll"]
sim = Simulation("COREWIND", "morro_bay_in_situ.yaml")

# Both of these parameters are True by default for convenience
sim.run(create_metrics=True, save_metrics_inputs=True)

# Load the metrics data
fpath = sim.env.metrics_input_fname.parent
fname = sim.env.metrics_input_fname.name
metrics = Metrics.from_simulation_outputs(fpath, fname)

# Delete the log files now that they're loaded in
sim.env.cleanup_log_files()

# Alternatively, in this case because the simulation was run, we can use the
# following for convenience convenience only
metrics = sim.metrics
```

(metrics-demo:common-parameters)=
## Common Parameter Explanations

Before diving into each and every metric, and how they can be customized, it is worth
noting some of the most common parameters used throughout, and their meanings to reduce
redundancy. The varying output forms are demonstrated in the
[availability](metrics-demo:availability) section below.

### `frequency`

  project
  : Computed across the whole simulation, with the resulting `DataFrame` having an empty
    index.

  annual
  : Summary of each year in the simulation, with the resulting `DataFrame` having "year"
    as the index.

  monthly
  : Summary of each month of the year, aggregated across years, with the resulting
    `DataFrame` having "month" as the index.

  month-year
  : computed on a month-by-year basis, producing the results for every month of the
    simulation, with the resulting `DataFrame` having "year" and "month" as the index.

### `by`

  windfarm
  : Aggregated across all turbines, with the resulting `DataFrame` having only
    "windfarm" as a column

  turbine
  : Computed for each turbine, with the resulting `DataFrame` having a column for each
    turbine

(metrics-demo:availability)=
## Availability

There are two methods to produce availability, which have their own function calls:

- energy: actual power produced divided by potential power produced
  - {py:meth}`wombat.core.post_processor.Metrics.production_based_availability`
- time: The ratio of all non-zero hours to all hours in the simulation, or the
  proportion of the simulation where turbines are operating
  - {py:meth}`wombat.core.post_processor.Metrics.time_based_availability`

Here, we will go through the various input definitions to get time-based availability data as both methods use the same inputs, and provide outputs in the same format.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters) options: "project",
  "annual", "monthly", and "month-year"
- `by`, as explained [above](metrics-demo:common-parameters) options: "windfarm" and
  "turbine"

Below is a demonstration of the variations on `frequency` and `by` for
`time_based_availability`.

```{code-cell} ipython3
:tags: ["output_scroll"]
style(metrics.time_based_availability(frequency="project", by="windfarm"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
style(metrics.production_based_availability(frequency="project", by="windfarm"))
```

Note that in the two above examples, that the values are equal. This is due to the fact
that the example simulation does not have any operating reduction applied to failures,
unless it's a catastrophic failure, so there is no expected difference.

```{code-cell} ipython3
:tags: ["output_scroll"]
# Demonstrate the by turbine granularity
style(metrics.time_based_availability(frequency="project", by="turbine"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Demonstrate the annualized outputs
style(metrics.time_based_availability(frequency="annual", by="windfarm"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Demonstrate the month aggregations
style(metrics.time_based_availability(frequency="monthly", by="windfarm"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Demonstrate the granular monthly reporting
style(metrics.time_based_availability(frequency="month-year", by="windfarm"))
```

### Plotting Availability

As of v0.9, the ability to plot the wind farm and turbine availability has been enabled
as an experimental feature. Please see the [plotting API documentation](plotting) for more details.

```{code-cell} ipython3
:tags: ["output_scroll"]
# Demonstrate the granular monthly reporting
plot.plot_farm_availability(sim=sim, which="energy", farm_95_CI=True)
```

(metrics-demo:cf)=
## Capacity Factor

The capacity factor is the ratio of actual (net) or potential (gross) energy production
divided by the project's capacity. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.capacity_factor`.

**Inputs**:

- `which`
  - "net": net capacity factor, actual production divided by the plant capacity
  - "gross": gross capacity factor, potential production divided by the plant capacity
- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by`, as explained [above](metrics-demo:common-parameters), options: "windfarm" and
  "turbine"

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
net_cf = metrics.capacity_factor(which="net", frequency="project", by="windfarm").values[0][0]
gross_cf = metrics.capacity_factor(which="gross", frequency="project", by="windfarm").values[0][0]
print(f"  Net capacity factor: {net_cf:.2%}")
print(f"Gross capacity factor: {gross_cf:.2%}")
```

(metrics-demo:task-completion)=
## Task Completion Rate

The task completion rate is the ratio of tasks completed aggregated to the desired
`frequency`. It is possible to have a >100% completion rate if all maintenance and
failure requests submitted in a time period were completed in addition to those that
went unfinished in prior time periods. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.task_completion_rate`.

**Inputs**:

- `which`
  - "scheduled": scheduled maintenance only (classified as maintenance tasks in inputs)
  - "unscheduled": unscheduled maintenance only (classified as failure events in inputs)
  - "both": Combined completion rate for all tasks
- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
scheduled = metrics.task_completion_rate(which="scheduled", frequency="project").values[0][0]
unscheduled = metrics.task_completion_rate(which="unscheduled", frequency="project").values[0][0]
combined = metrics.task_completion_rate(which="both", frequency="project").values[0][0]
print(f"  Scheduled Task Completion Rate: {scheduled:.2%}")
print(f"Unscheduled Task Completion Rate: {unscheduled:.2%}")
print(f"    Overall Task Completion Rate: {combined:.2%}")
```

(metrics-demo:equipment-costs)=
## Equipment Costs

Sum of the costs associated with a simulation's servicing equipment, which excludes
materials, downtime, etc. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.equipment_costs`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by_equipment`
  - `True`: Aggregates all equipment into a single cost
  - `False`: Computes for each unit of servicing equipment

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project total at the whole wind farm level
style(metrics.equipment_costs(frequency="project", by_equipment=False))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals at servicing equipment level
style(metrics.equipment_costs(frequency="project", by_equipment=True))
```

(metrics-demo:utilization-rate)=
## Service Equipment Utilization Rate

Ratio of days when the servicing equipment is in use (not delayed for a whole day due to
either weather or lack of repairs to be completed) to the number of days it's present in
the simulation. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.service_equipment_utilization`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project"
  and "annual"

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals
style(metrics.service_equipment_utilization(frequency="project"))
```

(metrics-demo:vessel-crew-hours)=
## Vessel-Crew Hours at Sea

The number of vessel hours or crew hours at sea for offshore wind power plant
simulations. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.vessel_crew_hours_at_sea`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by_equipment`
  - `True`: Aggregates all equipment into a single cost
  - `False`: Computes for each unit of servicing equipment
- `vessel_crew_assumption`: A dictionary of vessel names
  (`ServiceEquipment.settings.name`, but also found at `Metrics.service_equipment_names`)
  and the number of crew onboard at any given time. The application of this assumption
  transforms the results from vessel hours at sea to crew hours at sea.

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project total, not broken out by vessel
style(metrics.vessel_crew_hours_at_sea(frequency="project", by_equipment=False))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Annual project totals, broken out by vessel
style(metrics.vessel_crew_hours_at_sea(frequency="annual", by_equipment=True))
```

(metrics-demo:n-tows)=
## Number of Tows

The number of tows performed during the simulation. If tow-to-port was not used in the
simulation, a DataFrame with a single value of 0 will be returned. For further
documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.number_of_tows`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by_tug`
  - `True`: Computed for each tugboat (towing vessel)
  - `False`: Aggregates all the tugboats
- `by_direction`
  - `True`: Computed for each direction a tow was performed (to port or to site)
  - `False`: Aggregates to the total number of tows

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project Total
# NOTE: This example has no towing, so it will return 0
style(metrics.number_of_tows(frequency="project"))
```

(metrics-demo:labor-costs)=
## Labor Costs

Sum of all labor costs associated with servicing equipment, excluding the labor defined
in the fixed costs, which can be broken out by type. For further documentation, see the
API docs here: {py:meth}`wombat.core.post_processor.Metrics.labor_costs`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by_type`
  - `True`: Computed for each labor type (salary and hourly)
  - `False`: Aggregates all the labor costs

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project total at the whole wind farm level
total = metrics.labor_costs(frequency="project", by_type=False)
print(f"Project total: ${total.values[0][0] / metrics.project_capacity:,.2f}/MW")
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals for each type of labor
style(metrics.labor_costs(frequency="project", by_type=True))
```

(metrics-demo:equipment-labor-costs)=
## Equipment and Labor Costs

Sum of all labor and servicing equipment costs, excluding the labor defined in the fixed
costs, which can be broken out by each category. For further documentation, see the API
docs here: {py:meth}`wombat.core.post_processor.Metrics.equipment_labor_cost_breakdown`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by_category`
  - `True`: Computed for each unit servicing equipment and labor category
  - `False`: Aggregated to the sum of all costs

`reason` definitions:

- Maintenance: routine maintenance, or events defined as a
  {py:class}`wombat.core.data_classes.Maintenance`
- Repair: unscheduled maintenance, ranging from inspections to replacements, or events
  defined as a {py:class}`wombat.core.data_classes.Failure`
- Mobilization: Cost of mobilizing servicing equipment
- Crew Transfer: Costs incurred while crew are transferring between a turbine or
  substation and the servicing equipment
- Site Travel: Costs incurred while transiting to/from the site and while at the site
- Weather Delay: Any delays caused by unsafe weather conditions
- No Requests: Equipment and labor is active, but there are no repairs or maintenance
  tasks to be completed
- Not in Shift: Any time outside the operating hours of the wind farm (or the
  servicing equipment's specific operating hours)

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals
style(metrics.equipment_labor_cost_breakdowns(frequency="project", by_category=False))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals by each category
style(metrics.equipment_labor_cost_breakdowns(frequency="project", by_category=True))
```

(metrics-demo:emissions)=
## Emissions

Emissions (tons or other provided units) of all servicing equipment activity, except
overnight waiting periods between shifts. For further documentation, see the API docs
here: {py:meth}`wombat.core.post_processor.Metrics.emissions`.

**Inputs**:

- `emissions_factors`: Dictionary of servicing equipment names and the emissions per
  hour of the following activities: `transit`, `maneuvering`, `idle at site`, and
  `idle at port`, where port is stand-in for wherever the servicing equipment might be
  based when not at site.
- `maneuvering_factor`: The proportion of transit time that can generally be associated
  with positioning servicing, by default 10%.
- `port_engine_on_factor`: The proportion of the idling at port time when the engine is
  running and producing emissions, by default 25%.

```{code-cell} ipython3
# Create the emissions factors, in tons per hour
emissions_factors = {
    "Crew Transfer Vessel 1": {
        "transit": 4,
        "maneuvering": 3,
        "idle at site": 0.5,
        "idle at port": 0.25,
    },
    "Field Support Vessel": {
        "transit": 6,
        "maneuvering": 4,
        "idle at site": 1,
        "idle at port": 0.5,
    },
    "Heavy Lift Vessel": {
        "transit": 12,
        "maneuvering": 7,
        "idle at site": 1,
        "idle at port": 0.5,
    },
    "Diving Support Vessel": {
        "transit": 4,
        "maneuvering": 7,
        "idle at site": 0.2,
        "idle at port": 0.2,
    },
    "Anchor Handling Vessel": {
        "transit": 4,
        "maneuvering": 3,
        "idle at site": 1,
        "idle at port": 0.25,
    },
}

# Add in CTVs 2 through 7
for i in range(2, 8):
    emissions_factors[f"Crew Transfer Vessel {i}"] = emissions_factors[f"Crew Transfer Vessel 1"]

style(metrics.emissions(emissions_factors=emissions_factors, maneuvering_factor=0.075, port_engine_on_factor=0.20))
```

(metrics-demo:component-costs)=
## Component Costs

All the costs associated with maintenance and failure events during the simulation,
including delays incurred during the repair process, but excluding costs not directly
tied to a repair. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.component_costs`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by_category`
  - `True`: Computed across each cost category
  - `False`: Aggregated to the sum of all categories
- `by_action`
  - `True`: Computed by each of "repair", "maintenance", and "delay", and is included in
    the MultiIndex
  - `False`: Aggregated as the sum of all actions

`action` definitions:

- maintenance: routine maintenance
- repair: unscheduled maintenance, ranging from inspections to replacements
- delay: Any delays caused by unsafe weather conditions or not being able to finish a
  process within a single shift

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals by component
style(metrics.component_costs(frequency="project", by_category=False, by_action=False))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals by each category and action type
style(metrics.component_costs(frequency="project", by_category=True, by_action=True))
```

(metrics-demo:fixed-costs)=
## Fixed Cost Impacts

Computes the total costs of the fixed costs categories. For further documentation, see
the definition docs, here: {py:class}`wombat.core.data_classes.FixedCosts`, or the API
docs here: {py:meth}`wombat.core.post_processor.Metrics.fixed_costs`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `resolution` (also, demonstrated below)
  - "high": Computed across the most granular cost levels
  - "medium": Computed for each general cost category
  - "low": Aggregated to a single sum of costs

```{code-cell} ipython3
:tags: ["output_scroll"]
pprint(metrics.fixed_costs.hierarchy)
```

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals at the highest level
# NOTE: there were no fixed costs defined in this example, so all values will be 0, so
#       this will just be demonstrating the output format
style(metrics.project_fixed_costs(frequency="project", resolution="low"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals at the medium level
style(metrics.project_fixed_costs(frequency="project", resolution="medium"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals at the lowest level
style(metrics.project_fixed_costs(frequency="project", resolution="high"))
```

(metrics-demo:opex)=
## OpEx

Computes the total cost of all operating expenditures for the duration of the
simulation, including fixed costs. For further documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.opex`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"

- `by_category`
  - `True` shows the port fees, fixed costs, labor costs, equipment costs, and materials
    costs in addition the total OpEx
  - `False` shows only the total OpEx

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
style(metrics.opex("annual"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
style(metrics.opex("annual", by_category=True))
```

(metrics-demo:process-times)=
## Process Times

Computes the total number of hours spent from repair request submission to completion,
performing repairs, and the number of request for each repair category. For further
documentation, see the API docs here:
{py:meth}`wombat.core.post_processor.Metrics.process_times`.

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
style(metrics.process_times())
```

(metrics-demo:power-production)=
## Power Production

Computes the total power production for the wind farm. For further documentation, see
the API docs here: {py:meth}`wombat.core.post_processor.Metrics.power_production`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `by`, as explained [above](metrics-demo:common-parameters) options: "windfarm" and "turbine"
- `units`
  - "kwh": kilowatt-hours (kWh)
  - "mwh": megawatt-hours (MWh)
  - "gwh": gigawatt-hours (GWh)

**Example Usage**:

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals, in kWh, at the wind farm level
style(metrics.power_production(frequency="project", by="windfarm", units="kwh"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals, in MWh, at the wind farm level
style(metrics.power_production(frequency="project", units="mwh"))
```

```{code-cell} ipython3
:tags: ["output_scroll"]
# Project totals, in GWh, at the wind farm level
style(metrics.power_production(frequency="project"))
```

(metrics-demo:npv)=
## Net Present Value

Calcualtes the net present value (NPV) for the project, as
$NPV = (Power * OfftakePrice - OpEx) / (1 + DiscountRate)$.

For further documentation, see the API docs here: {py:meth}`wombat.core.post_processor.Metrics.npv`.

**Inputs**:

- `frequency`, as explained [above](metrics-demo:common-parameters), options: "project",
  "annual", "monthly", and "month-year"
- `discount_rate`: The rate of return that could be earned on alternative investments,
  by default 0.025.
- `offtake_price`: Price of energy, per MWh, by default 80.

```{code-cell} ipython3
:tags: ["output_scroll"]
style(metrics.opex("annual"))
```
