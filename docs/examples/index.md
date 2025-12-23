
# User Guide

This page provides an overview of what WOMBAT is currently able to model, broken down by
general category. Following this, there are separate pages for how this is done as
demonstrated through example notebooks. To fully follow the particulars of each example,
it is recommended to see how each model's configuration files are composed.

For thorough explanations of the design and implementation ethos of the model, please
see our NREL Technical Report: https://www.osti.gov/biblio/1894867 {cite}`osti_1894867`,
which was published alongside v0.5.1, so some functionality has been updated.

## Feature Overview

For a complete and detailed description of the functionality provided in WOMBAT, it is
recommended to read the API documentation. However, that is quite a task, so below is
a shortlist of the core functionality that WOMBAT can provide.

### Post Processing and the Simulation API

- The [`Simulation` class](simulation-api) and its
  [`Configuration`](simulation-api:config) allow users unfamiliar with
  Python to run the simulation library with minimal code. This was the primary design
  choice for the model: to create a configuration-based model to enable users with nearly
  any level of Python exposure.
- The [`Metrics`](simulation-api:metrics) class provides a straightforward interface for
  computing a wide variety of operational summary statistics from performance to wind
  power plant financials.

### Environmental Considerations

- Wind speed (m/s) and wave height (m) for the duration of a simulation (user-provided,
  hourly profile)
- Reduced speed periods for animal migrations. This is primarily an offshore-focused
  feature, but can be defined at the simulation, port, or servicing equipment level
  using the following three variables: `reduced_speed_start`, `reduced_speed_end`, and
  `reduced_speed` (km/hr). This translates to a maximum speed of `reduced_speed` being
  enforced between the starting and ending dates each year of the simulation. For more
  details, see the documentation pages for the
  [environment](core:environment), [port](core:port),
  [unscheduled servicing equipment](types:service-equipment:unscheduled), or
  [scheduled servicing equipment](types:service-equipment:scheduled).
- Prohibited operation periods. This dictates periods of time when a site or port may
  be inaccessible, or when a piece of servicing equipment should not be allowed to be
  mobilized/dispatched. Similar to the speed reduction periods, `non_operational_start`
  and `non_operational_end` dictate an annual date range for when servicing equipment
  can't be active, or a port or the whole site is inaccessible. The same documentation
  pages as above can be referenced for more details.
- Generalized maintenance start date. This enables the first occurrence for all
  maintenance events to be set universally, if not specifically defined. If the date
  is prior to the start of the simulation, it merely sets the cadence. The ability to
  set a maintenance starting date allows for the staggering of the weather timeseries
  and maintenance timing. For instance, a weather profile starting in January for the
  Gulf of Maine, would have annual maintenance occurring in the middle of winter, likely
  causing many unnecessary weather delays in accessing the site, so the cadence can
  now start in the late spring or summer to ensure there are ideal weather conditions.
  This feature is made available through the
  [`Configuration.maintenance_start`](simulation-api:config) field,

### System and Subassembly Modeling

Each substation, turbine, and cable operate with the same core logic of
[maintenance](types:maintenance:scheduled) and [failure](types:maintenance:unscheduled)
events. The following will break down the differences between each of these systems'
modeling assumptions as well as the basic operations of the maintenance and failure models.

- Maintenance
  - `frequency`: based on an amount of time between events and a given starting date.
  - `frequency_basis`: allows the input of frequency to be configured in days, months,
    or years.
  - `start_date`: enables the first occurrence of a maintenance event to be offset
    from the weather profile, or even well after the start date of a simulation.
  - `operation_reduction`: the percentage of degradation the triggering of this event
    cause
- Failure
  - Randomly samples from a Weibull distribution to determine the time to the next
    failure
  - `operation_reduction`: the percentage of degradation the triggering of this event
    will cause
  - `replacement`: if `True`, then all preceding failures and maintenance tasks are
    cancelled and the time until the next event is reset because a replacement is
    required, otherwise, each successive failure is added to the repair queue
- Commonalities between Substations, Turbines, and Cables
  - Maintenance and Failures compose the actual "model"
  - `operating_level` provides the real degradation of a model, including if it is turned
    off for servicing
  - `operating_level_wo_servicing` provides the operating level as if there were no
    ongoing operations
- Substations
  - Any failure or maintenance with an `operation_reduction` of 1 will turn off every
    upstream connected (and modeled) cable and turbine
  - For substations that are connected by an export cable this, the upstream connections
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
- Electrolyzers (new in v0.11)
  - Similar to both the turbine model, except that they are only allowed to be connected
    to a substation. Electrolyzer downtime does not impact the rest of the farm.
  - Requires at least 1 turbine with a power curve and 1 substation to enable H2
    production. The turbine and substation may be modeled without any failures as a
    method to model only an electrolyzer.
  - **NOTE**: in a future version (likely v0.12) there will be a simplified
    way to model only an electrolyzer, and provide an external power profile for a more
    accurate assessment of the capacity factor and hydrogen production.

### Repair Modeling

- Repair Management: see [here](core:repair-manager) for complete details
  - Generally, repairs and maintenance tasks operate on a first-in, first-out basis with
    higher severity level `Maintenance.level` and `Failure.level` being addressed first.
  - Servicing equipment, however, can specify if they operate on a "severity" or "turbine"
    basis that prioritizes focusing on either the highest severity level first, or
    a single system first, respectively.
- Servicing Equipment: see [here](core:service-equipment) for complete details
  - Can either be modeled on a scheduled basis, such as a year-round schedule for onsite
    equipment or equipment with an annual visit schedule during safe weather, or on
    an unscheduled basis with a threshold for number of submitted repair requests or
    farm operating level.
  - Mobilizations can be modeled with a fixed number of days to get to site and a flat
    cost for both scheduled and unscheduled servicing equipment. For scheduled equipment,
    the mobilization is scheduled so that the equipment arrives at the site for the
    start of it's first scheduled day.
  - Mooring disconnections and reconnections operate without considering shift timing
    to ensure that these (typically) very long processes can find an optimal weather
    window within the simulation. These typically only occur with tugboats.
  - A wide range of generalized capabilities can be specified for various modeling
    scenarios. It's important to note that outside of the "TOW" designation, there are
    no specific implementations of functionality.
    - CTV: crew transfer vessel/vehicle
    - OFS: offsite equipment
    - SCN: small crane (i.e., field support vessel)
    - MCN: medium crane
    - LCN: large crane (i.e., heavy lift vessel)
    - CAB: cabling vessel/vehicle
    - RMT: remote reset (no actual equipment BUT no special implementation)
    - DRN: drone
    - DSV: diving support vessel
    - VSG: vessel support group
    - TOW: tugboat and support vessel (triggers the tow-to-port model)
    - AHV: anchor handling vessel (special case that can be modeled from the port or
      main simulation configuration)
  - Operating limits can be applied for both transiting and repair logic to ensure site
    safety is considered in the model.
- Ports
  :::{note}
  Major improvements as of v0.13
  :::
  - Currently only used for tow-to-port repairs or tugboat based repairs, which adds the
    following additional capabilities:
    - TOW: tugboat or towing equipment
    - AHV: anchor handling vessel (tugboat that doesn't trigger tow-to-port)
  - Port access fees can be modeled in the `FixedCosts` module, or applied as a monthly
    fee using either the `annual_fee` or `monthly_fee` input.
  - Port usage fees are applied when any turbine is at the port for repairs, and can
    set using the `daily_use_fee` parameter.
  - Tugboats are mobilized, and will stay at port for the duration of their
    `charter_days` parameter.
  - Tugboats are called out on the first request from the requesting system,
    regardless of the user encoding.
  - All subsequent repair and maintenance requests following a "TOW" request will not
    be addressed until the system is at the port. This can create a significant
    backlog of repairs if there are not enough tugboats, crews, or active turbine
    slots encoded at the port.
  - See the [API docs](core:port) for more details

## Examples and Validation Work

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)

Below are a few examples to get started, for users interested in the validation work in
the [code-to-code comparison presentations](presentations:code-comparison),
the notebooks generating [the most up-to-date results can be found in the main repository](https://github.com/WISDEM/WOMBAT/examples/), where there is a separate analysis
for the
[{cite:t}`dinwoodie2015reference` comparison](https://github.com/WISDEM/WOMBAT/blob/main/examples/dinwoodie_validation.ipynb),
and for the [{cite:t}`smart2016iea` comparison](https://github.com/WISDEM/blob/main/WOMBAT/examples/iea_26_validation.ipynb).

<!-- ```{include} ../../library/default/README.md``` -->

## Default Data

As of version 0.13, a default reference data set has been made available for users. Simply specify
a simulation as `Simulation.from_config("DEFAULT", "osw_fixed.yaml")` or
`Simulation.from_config("DEFAULT", "osw_floating.yaml")` to use either of the fixed-bottom or floating
offshore wind data sets, respectively. See the `examples/COWER_om_workflow.ipnyb` for more details.

The default library provides a validated, ready-to-use data set for fixed and floating offshore wind,
and an experimental land-based data set. For all three, users can use the pre-configured base models
or use them as a starting point for building custom models. Example results can be seen in
`examples/default_data_demonstration.ipynb` or [online](default_data_demonstration.md)

### Overview

#### Offshore

The **default library** provides a consistent set of cost and performance inputs for offshore wind
energy systems, standardized to 2024 USD. This library supports reproducible analyses in the
*Cost of Wind Energy Review (COWER): 2025 Edition* (forthcoming), including both fixed-bottom and
floating offshore wind cases.

Unlike other datasets in this repository, which largely reflect publicly available sources, this
dataset incorporates **internal adjustments and harmonizations** to align with research-focused
scenarios for NREL products that may require outputs to be in 2024 USD, like for example, the Annual
Technology Baseline, or the Cost of Wind Energy Review.

#### Land-Based

The default library provides an experimental set of land-based data that is hybridized from a series
of incomplete sources of offshore and onshore costs, and scaling factors between them. It is highly
recommended to merely use this data as a starting point for any analysis work.

### Core Assumptions

#### Offshore

Material costs for repairs and replacements, as well as failure data, are sourced from
{cite:t}`corewind_d61_2020, corewind_d42_2021`. Fixed cost data is primarily derived from
{cite:t}`bvg_guide_2023, bvg_guide_2025`, while vessel day rate and mobilization assumptions are
compiled from a range of public sources. For detailed reference information, please contact Daniel
Mulas Hernando (Daniel.MulasHernando@nrel.gov) to request access to WOMBAT_cost_history.xlsx.

- **Cost Year:** All monetary values are standardized to **2024 USD**.
- **Data Integration:** Inputs were consolidated from multiple public sources and internal records (`WOMBAT_cost_history.xlsx`), with historical exchange rates and inflation applied to transform costs to 2024 USD.
- **Technology Coverage:** Includes representative inputs for offshore (fixed-bottom and floating) technologies.

#### Land-Based

Failure rates, repair times, and materials costs not provided in {cite:t}`corewind_d42_2021` are
largely taken from {cite:t}`carroll2016failure` with some custom substitutions or modifications. All
failure rates are converted to a Weibull scale factor. Using {cite:t}`carroll2016failure` we adjust
offshoreWeibull scale factors for minor repairs for the generator, blades, pitch system, yaw system,
and drive train (wind-affected subassemblies) by a factor of 5.912, for non-wind-affected
subassemblies a factor of 1.217 is applied, and for all major repairs and replacements, a factor of
2.432 is applied. These scaling factors are primarily derived from {cite:t}`carroll2016failure`

All costs are rescaled from the 12 MW baseline to a 3.5 MW baseline using the COWER 2025 turbine
CapEx figures of \$1,117/kw-yr for onshore and \$1,770/kw-yr for offshore, a 0.6311 scaling factor.

Additional subassemblies are derived from {cite:t}`carroll2016failure` and scaled using the above
assumptions. Fixed costs are derived from {cite:t}`WISER201946`, with labor coming supplemented by
{cite:t}`osti_1995015`, and equipment costs come from LandBOSSE and internal source adjustments.

### Intended Use

- Serve as a **baseline input** for replicable analyses of the offshore fixed and floating *COWER-2025* results.
- Serve as a **baseline input** for the development of land-based analyses.
- Support **scenario development** and **sensitivity analyses** exploring the impact of cost evolution, operational performance, and logistics assumptions.

### Reproducibility

The accompanying notebook, **`examples/COWER_om_workflow.ipynb`**, demonstrates how to replicate O&M results from the *Cost of Wind Energy Review: 2025 Edition*. It runs 50 simulations per case and summarizes mean and standard deviation results to identify sources of variability within cost components.

#### Notes

- `WOMBAT_cost_history.xlsx` is an internal NREL document. For questions, contact **Daniel.MulasHernando@nrel.gov**.
- This dataset should be treated as a **scenario-based reference**, not as purely empirical or historical data.

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
