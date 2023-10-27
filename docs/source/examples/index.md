
Reference
=========

This page provides an overview of what WOMBAT is currently able to model, broken down by
general category. Following this, there are separate pages for how this is done as
demonstrated through example notebooks. To fully follow the particulars of each example,
it is recommended to see how each model's configuration files are composed.

For thorough explanations of the design and implementation ethos of the model, please
see our NREL Technical Report: https://www.osti.gov/biblio/1894867, which was published
alongside v0.5.1, so some functionality has been updated.
## Feature Overview

For a complete and detailed description of the functionality provided in WOMBAT, it is
recommended to read the API documentation. However, that is quite a task, so below is
a short listing of the core functionality that WOMBAT can provide.

### Post Processing and the Simulation API

- The [`Simulation` class](../API/simulation_api.md#simulation-api) and its
  [`Configuration`](../API/simulation_api.md#configuration) allow users unfamiliar with
  Python to run the simulation library with minimal code. This was the primary design
  choice for the model: to create a configuration-based model to enable users with nearly
  any level Python exposure.
- The [`Metrics`](../API/simulation_api.md#metrics-computation) class provides a
  straightfoward interface for computing a wide variety of operational summary statistics
  from performance to wind power plant financials.

### Environmental Considerations

- Windspeed (m/s) and wave height (m) for the duration of a simulation (user-provided,
  hourly profile)
- Reduced speed periods for animal migrations. This is primarily an offshore-focused
  feature, but can be defined at the simulation, port, or servicing equipment level
  using the following three variables: `reduced_speed_start`, `reduced_speed_end`, and
  `reduced_speed` (km/hr). This translates to a maximum speed of `reduced_speed` being
  enforced between the starting and ending dates each year of the simulation. For more
  details, see the documentation pages for the
  [environment](../API/simulation.md#environment), [port](../API/simulation.md#port),
  [unshcheduled servicing equipment](../API/types.md#unscheduled-service-equipment), or
  [scheduled servicing equipment](../API/types.md#scheduled-service-equipment).
- Prohibited operation periods. This dictates periods of time where a site or port may
  be inaccessible, or when a piece of servicing equipment should not be allowed to be
  mobilized/dispatched. Similar to the speed reduction periods, `non_operational_start`
  and `non_operational_end` dictate an annual date range for when servicing equipment
  can't be active, or a port or the whole site is inaccessible. The same documenation
  pages as above can be referenced for more details.

### System and Subassembly Modeling

Each substation, turbine, and cable operate with the same core logic of
[maintenance](../API/types.md#maintenance-tasks) and [failure](../API/types.md#failures)
events. The following will break down the differences between each of these systems'
modeling assumptions as well as the basic operations of the maintenance and failure models.

- Maintenance
  - `frequency`: based on a fixed number of days between an event
  - `operation_reduction`: the percentage of degradation the triggering of this event
    cause
- Failure
  - Randomly samples from a Weibull distribution to determine the next failure
  - `operation_reduction`: the percentage of degradation the triggering of this event
    cause
  - `replacement`: if `True`, then all preceeding failures and maintenance tasks are
    cancelled and the time until the next event is reset because a replacement is
    required, otherwise, each successive failure is added to the repair queue
- Commonalities between Substations, Turbines, and Cables
  - Maintenance and Failures compose the actual "model"
  - `operating_level` provides the real degradation of a model, including if it is turned
    off for servicing
  - `operating_level_wo_servicing` provides the operting level as if there were no
    ongoing operations to know the potential operating operating
- Substations
  - Any failure or maintenance with an `operation_reduction` of 1 will turn off every
    upstream connected (and modeled) cable and turbine
  - For subations that are connected by an export cable this, the upstream connections
    made from the export cable will not be considered upstream connections and therefore
    shutdown when a downstream substation has a failure
- Array cables
  - Similar to substations, any failure or maintenance with an `operation_reduction` of
    1 will turn off every connected upstream cable and turbine on the same string
- Export Cables
  - Similar to array cables, any failure or maintenance with an `operation_reduction` of
    1 will turn off every connected upstream substation and string of cable(s) and
    turbine(s)
  - As noted in the substation section, export cables connecting substations act
    independently and not as a string of connected systems
  - The final export cable connecting a substation to the interconnection point can
    however shut down the entire wind farm

### Repair Modeling

- Repair Management: see [here](../API/simulation.md#repair-management) for complete details
  - Generally, repairs and maintenance tasks operate on a first-in, first-out basis with
    higher severity level `Maintenance.level` and `Failure.level` being addressed first.
  - Servicing equipment, however, can specify if they operate on a "severity" or "turbine"
    basis that prioritizes focusing on either the highest serverity level first, or
    a single system first, respectively.
- Servicing Equipment: see [here](../API/simulation.md#servicing-equipment) for complete details
  - Can either be modeled on a scheduled basis, such as a year-round schedule for onsite
    equipment or equipment with an annual visit schedule during safe weather, or on
    an unscheduled basis with a threshold for number of submitted repair requests or
    farm operating level.
  - Mobilizations can be modeled with a fixed number of days to get to site and a flat
    cost for both scheduled and unscheduled servicing equipment. For scheduled equipment,
    the mobilization is scheduled so that the equipment arrives at the site for the
    start of it's first scheduled day.
  - A wide range of generalized capabilities can be specified for various modeling
    scenarios.
    - RMT: remote (no actual equipment BUT no special implementation)
    - DRN: drone
    - CTV: crew transfer vessel/vehicle
    - SCN: small crane (i.e., field support vessel)
    - LCN: large crane (i.e., heavy lift vessel)
    - CAB: cabling vessel/vehicle
    - DSV: diving support vessel
  - Operating limits can be applied for both transiting and repair logic to ensure site
    safety is considered in the model.
- Ports
  - Currently only used for tow-to-port repairs or tugboat based repairs, which adds the
    following additional capabilities:
    - TOW: tugboat or towing equipment
    - AHV: anchor handling vessel (tugboat that doesn't trigger tow-to-port)
  - See the [API docs](../API/simulation.md#port) for more details

## Examples and Validation Work

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)

Below are a few examples to get started, for users interested in the validation work in
the [code-to-code comparison presentations](../presentations.md#code-to-code-comparison),
the notebooks generating [the most up-to-date results can be found in the main repository](https://github.com/WISDEM/WOMBAT/examples/), where there is a separate analysis
for the
[Dinwoodie, et. al, 2015 comparison](https://github.com/WISDEM/WOMBAT/blob/main/examples/dinwoodie_validation.ipynb),
and for the [IEA Task 26, 2016 comparison](https://github.com/WISDEM/blob/main/WOMBAT/examples/iea_26_validation.ipynb).

```{toctree}
:maxdepth: 1
:caption: Examples

how_to
strategy_demonstration
metrics_demonstration
examples_reference
```
