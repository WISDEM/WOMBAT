---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How To Use WOMBAT

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)

This tutorial will walk through the setup, running, and results stages of a WOMBAT
simulation while providing background information about how each component is related.

A Jupyter notebook of this tutorial can be run from `examples/how_to.ipynb` locally, or
through [binder](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples).

## Imports

The following code block demonstrates a typical setup for working with WOMBAT and
running analyses.

```{code-cell} ipython3
from time import perf_counter  # timing purposes only

import numpy as np
import pandas as pd

from wombat import Simulation
from wombat.core.library import load_yaml, DINWOODIE

# Seed the random variable for consistently randomized results
np.random.seed(0)

# Improve the legibility of DataFrames
pd.set_option("display.float_format", '{:,.2f}'.format)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
```

## Defining the Simulation

The following will demonstrate the required information to run a simulation. For the
purposes of this tutorial, we'll be working with the data under
`library/code_comparison/dinwoodie` in the Github repository, and specifically the base
case.

One important item to note is that the library structure is enforced within the code so all
data must be placed in the appropriate locations in your analysis' library as follows:

```{warning}
As of v0.6, the following structure will be adopted to mirror the format of the
[ORBIT library structure](https://github.com/WISDEM/ORBIT/blob/master/ORBIT/core/library.py#L7-L24)
to increase compatibility between similar libraries.

As of v0.9, the library structure shown below is the only one that will work.
```

To help users convert to the new structure, the following method is provided to create
the required folder structure for users.

```{code-block} python

from wombat import create_library_structure  # located in wombat.core.library

new_library = "library"
create_library_structure(new_library)
```

The above method call will produce the below folder and subfolder structure.

```
<library>
  ├── project
    ├── config     <- Project-level configuration files
    ├── port       <- Port configuration files
    ├── plant      <- Wind farm layout files
  ├── cables       <- Export and Array cable configuration files
  ├── substations  <- Substation configuration files
  ├── turbines     <- Turbine configuration and power curve files
  ├── vessels      <- Land-based and offshore servicing equipment configuration files
  ├── weather      <- Weather profiles
  ├── results      <- The analysis log files and any saved output data
```

As a convenience feature you can import the provided validation data libraries as
`DINWOODIE` or `IEA_26` as is seen in the imports above, and a consistent path will be
enabled.

In practice, any folder location can be used so long as it follows the subfolder
structure provided here.

(how_to:configure:layout)=
### Wind Farm Layout

The wind farm layout is determined by a csv file, `dinwoodie/windfarm/layout.csv` in
this case. Below is a sample of what information is required and how to use each field,
followed by a visual example. It should be noted that none of the headings are
case-sensitive.

id (required)
: Unique identifier for the asset; no spaces allowed.

substation_id (required)
: The id field for the substation that the asset connects to; in the case that this is a
  substation, then this field should be the same as `id`; no spaces allowed.

name (required)
: A descriptive name for the turbine, if desired. This can be the same as `id`.

type (optional)
: One of "turbine" or "substation". This is required to accurately model a
  multi-substation wind farm. The base assumption is that a substation connects to
  itself as a means to model the export cable connecting to the interconnection point,
  however, this is not always the case, as substations may be connected through their
  export systems. Using this filed allows for that connection to be modeled accurately.

longitude (optional)
: The longitudinal position of the asset, can be in any geospatial reference; optional.

latitude (optional)
: The latitude position of the asset, can be in any geospatial reference; optional.

string (required)
: The integer, zero-indexed, string number for where the turbine will be positioned.

order (required)
: The integer, zero-indexed position on the string for where the turbine will be
  positioned.

distance (optional)
: The distance to the upstream asset; if this is calculated (input = 0), then the
  straight line distance is calculated using the provided coordinates (WGS-84 assumed).

subassembly (required)
: The file that defines the asset's modeling parameters.

upstream_cable (required)
: The file that defines the upstream cable's modeling parameters.

upstream_cable_name (optional)
: The descriptive name to give to the cable that will be used during logging. This
  enables users to use a single cable definition file while maintaining the naming
  conventions used for the wind farm being simulated.

```{note}
In the example below, there are a few noteworthy caveats that will set the stage for
later input reviews:
 - The cables are not modeled, which has a couple of implications
   - There only needs to be one string "0"
   - The cable specifications are required, even if not being modeled (details later)
 - longitude, latitude, and distance are all "0" because the spatial locations are not used
 - subassembly is all "vestas_v90.yaml", but having to input the turbine subassembly model
  means that multiple turbine types can be used on a windfarm.
    - This same logic applies to the upstream_cable so that multiple cable types can be
     used as appopriate.
```

<div style="overflow-y:auto;overflow-x:auto">

| id | substation_id | name | type | longitude | latitude | string | order | distance | subassembly | upstream_cable |
| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | :-- | :-- |
| OSS1 | OSS1 | OSS1 | substation | 0 | 0 |  |  |  | offshore_substation.yaml | export.yaml |
| S00T1 | OSS1 | S00T1 | turbine | 0 | 0 | 0 | 0 | 0 | vestas_v90.yaml | array.yaml |
| S00T2 | OSS1 | S00T2 | turbine | 0 | 0 | 0 | 1 | 0 | vestas_v90.yaml | array.yaml |
| S00T3 | OSS1 | S00T3 | turbine | 0 | 0 | 0 | 2 | 0 | vestas_v90.yaml | array.yaml |
| ... |
| S00T79 | OSS1 | S00T79 | turbine | 0 | 0 | 0 | 78 | 0 | vestas_v90.yaml | array.yaml |
| S00T80 | OSS1 | S00T80 | turbine | 0 | 0 | 0 | 79 | 0 | vestas_v90.yaml | array.yaml |
</div>

(how_to:configure:weather)=
### Weather Profile

The weather profile will broadly define the simulation range with its start and stop
points, though a narrower one can be used when defining the simulation (more later).

The following columns should exist in the data set with the provided guidelines.

datetime (required)
: A date and time stamp, any format.

windspeed (required)
: The hourly, mean winds peed, in meters per second at the time stamp.

waveheight (optional)
: The hourly, mean wave height, in meters, at the time stamp. If waves are not required,
    this can be filled with zeros, or be left out entirely.

Below, is a demonstration of what `weather/alpha_ventus_weather_2002_2014.csv` looks like.

| datetime | windspeed | waveheight |
| :-- | --: | --: |
| 1/1/02 0:00 | 11.75561096 | 1.281772405 |
| 1/1/02 1:00 | 10.41321252 | 1.586584315 |
| 1/1/02 2:00 | 8.959270788 | 1.725690828 |
| 1/1/02 3:00 | 9.10014808 | 1.680982063 |
| ... |
| 12/31/14 22:00 | 14.40838803 | 0.869625003 |
| 12/31/14 23:00 | 14.04563195 | 0.993031445 |

(how_to:configure:environment)=
### Environmental Considerations

In addition to using weather conditions for site characteristics, WOMBAT is able to
model environmental considerations where a port or site cannot be accessed, for example,
when silt builds up and water depths become too low to tow a turbine into the port.
There is also a feature for imposing maximum operating speeds, such as when there are
animal migrations and vessels must slow down to avoid collisions with endangered species.

Defining the `non_operational_start` and `non_operational_end` at either the servicing
equipment, environment, or port level allows for the creation of an annualized date
range spanning the length of the simulation where operations are not allowed to occur.
When defined at the environment level, all servicing equipment and a port, if defined,
will have this non-operational period applied, and if it's already existing, the more
conservative of the date ranges will be applied. When defined at the port level, all
associated servicing equipment (tugboats) will have the same inuring priority as when
defined at the environment level.

The same logic applies when defining the `reduced_speed_start` and `reduced_speed_end`
for defining when the operating speeds of servicing equipment are capped at the
`reduced_speed`. As is the case above, these variables can also be defined at the
servicing equipment level for further customization.

(how_to:configure:fixed-costs)=
### Fixed Costs

Please see the [`FixedCosts` API documentation](types:misc:fixed-costs) for
details on this optional piece of financial modeling.

For modeling a tow-to-port strategy that the port rental costs should be included in
this category, and not in the port configuration.

(how_to:configure:servicing-equipment)=
### Servicing Equipment

The servicing equipment control the actual repairs within a simulation, and as of v0.5,
there are four different repair strategies that can be used: scheduled, downtime-based
unscheduled, repair-based unscheduled, and tow-to-port. These are options are controlled
through the `capability` settings in each equipment's configuration in conjunction with
the `service_equipment` setting in the maintenance and failure configurations for each
subassembly.

For complete documentation of how the servicing equipment parameters are defined, please
see the [ServiceEquipmentData API documentation](types:service-equipment)

Below is a definition of the different equipment codes and their designations to show
the breadth of what can be simulated. These codes do not have separate operating models,
but instead allow the user to specify the types of operations the servicing equipment
will be able to operate on. This model should be aligned with the `service_equipment`
requirements in the subassembly failure and maintenance models.

RMT
: remote (no actual equipment BUT no special implementation), akin to remote resets

DRN
: drone, or potentially even helicopters by changing the costs

CTV
: crew transfer vessel/onsite truck

SCN
: small crane (i.e., field support vessel or cherry picker)

MCN
: medium crane (i.e., anything between a cherry picker and a crawler crane)

LCN
: large crane (i.e., heavy lift vessel or crawler crane)

CAB
: cabling-specific vessel/vehicle

DSV
: diving support vessel

TOW
: tugboat/towing (a failure with this setting will trigger a tow-to-port scenario where
the simulation's `Port` will dispatch the tugboats as needed)

AHV
: anchor handling vessel (this is a variation on the tugboat for mooring repairs that
will not tow anything between port and site, but operate at the site)

VSG
: vessel support group (used as a means for modeling a grouping of vessels used for a
sing repair operation)

Aside from the TOW and AHV capabilities there are no operations specific to each
capability. The remaining configurations of a servicing equipment such as equipment
rates, mobilization, labor, and operating limits will define the nature of its operation
in tandem with a failure's `time` field. So for a remote reset (RMT), there will be a
trivial equipment rate, if any, associated with it to account for the specific operations
or resetting the subassembly remotely. Similarly, a drone repair or inspection (DRN)
will not require any onboard crew, so labor will 0, or low if assuming an operator that
is unaccounted for in the site `FixedCosts`, but will require more time than a remote
reset in addition to a higher equipment cost.

In addition to a variety of servicing equipment types, there is support for
3 different equipment-level dispatch strategies, as described below. For a set of
example scenarios, please see the [strategy demonstration](strategy_demonstration.md).

scheduled
: dispatch servicing equipment for a specified date range each year

requests
: dispatch the servicing equipment once a `strategy_threshold` number of requests
  that the equipment can service has been reached

downtime
: dispatch the servicing equipment once the wind farm's operating level reaches the
  `strategy_threshold` percent downtime.

### The System Models

The actual assets on the windfarm such as cables, turbines, and substations, are
referred to as systems in WOMBAT, and each has their own individual model. Within each
of these systems, there are user-defined subassemblies (or components) that rely on
two types of repair models:

- maintenance: scheduled, fixed time interval-based maintenance tasks
- failures: unscheduled, Weibull distribution-based modeled, maintenance tasks

The subassemblies keys in the system YAML definition can be user-defined
to accommodate the varying language among industry groups and generations of wind
technologies. The only restrictions are the YAML keys must not contain any special
characters, and cannot be repeated

In the example below we show a `generator` subassembly with an annual service task and
frequent manual reset. For a thorough definition, please read the API
documentation of the [Maintenance](types:maintenance:scheduled) and
[Failure](types:maintenance:unscheduled) data classes. Note that the yaml definition below
specifies that maintenance tasks are in a bulleted list format and that failure
defintions require a dictionary-style input with keys to match the severity level of a
given failure. For more details on the complete subassembly definition, please visit the
[Subassembly API documentation](types:windfarm:subassembly).

```{important}
As of v0.10, failure configurations (`failures`) require a list-based definition. To
convert older cable, subastation, and turbine configuration failues, use the
library function `convert_failure_data` as shown in the
[helpers API documentation](importing-and-converting-from-old-versions).
```

```{code-block} yaml
generator:
  name: generator  # doesn't need to match the subassembly key that becomes System.id
  maintenance:
  - description: annual service
    time: 60
    materials: 18500
    service_equipment: CTV
    frequency: 365
    operation_reduction: 0  # default value
  failures:
    -
      scale: 0.1333
      shape: 1
      time: 3
      materials: 0
      service_equipment: CTV
      operation_reduction: 0.0
      replacement: False  # default value
      level: 1  # Note that the "level" value matches the key "1"
      description: manual reset
```

#### Substations

The substation model relies on two specific inputs, and one subassembly input (transformer).

capacity_kw
: The capacity of all turbines in the wind farm, neglecting any losses. Only needed if
a $/kw cost basis is being used.

capex_kw
: The $/kw cost of the machine, if not providing absolute costs.

Additional keys can be added to represent subassemblies, such as a transformer, in the
same format as the generator example above. Similarly, a user can define as man or as
few of the subassemblies as desired with their preferred naming conventions

The following is an example of substation YAML definition with no modeled subassemblies.

```{code-block} yaml
capacity_kw: 670000
capex_kw: 140
transformer:
  name: transformer
  maintenance:
    -
      description: n/a
      time: 0
      materials: 0
      service_equipment: CTV
      frequency: 0
  failures:
    -
      scale: 0
      shape: 0
      time: 0
      materials: 0
      service_equipment: [CTV]
      operation_reduction: 0
      level: 1
      description: n/a
```

#### Turbines

The turbine has the most to define out of the three systems in the wind farm model.
Similar to the substation, it relies mainly on the subassembly model with a few extra
parameters, as defined here:

capacity_kw
: The capacity of the system. Only needed if a $/kw cost basis is being used.

capex_kw
: The $/kw cost of the machine, if not providing absolute costs.

power_curve: file
: File that provides the power curve definition.

power_curve: bin_width
The desired interval, in m/s, between adjacent points on the power curve to be used for
power calculations.

The `windfarm/vestas_v90.yaml` data file provides the following definition in addition
to the maintenance and failure definitions that were shown previously.

```{code-block} yaml
capacity_kw: 3000
capex_kw: 1300
power_curve:
  file: vestas_v90_power_curve.csv
  bin_width: 0.5
```

The power curve input CSV requires the following two columns: `windspeed_ms` and
`power_kw` that should be defined using the wind speed for a bin, in m/s and the power
produced at that wind speed, in kW. The current method available for generating the
power curve is the IEC 61400-12-1-2 method for a wind-speed binned power curve. If there
is a need/desire for additional power curve methodologies, then
[please submit an issue on the GitHub](https://github.com/WISDEM/WOMBAT/issues)!

For an open source listing of a variety of land-based, offshore, and distributed wind
turbine power curves, please visit the
[NREL Turbine Models repository](https://github.com/NREL/turbine-models).

#### Cables

The array cable is the simplest format in that you only define a descriptive name,
and the maintenance and failure events as below. It should be noted that the scheme
is a combination of both the system and subassembly configuration.

For export cables that connect substations, as opposed to an interconnection, they will
not create dependencies because it is assumed they bypass the substation altogether,
and connect directly with the export system. This means that if there is a failure at a
downstream substation, the connecting export cable and its upstream turbine, substation,
and cable connections will continue to operate normally.

```{code-block} yaml
name: array cable
maintenance:
  -
    description: n/a
    time: 0
    materials: 0
    service_equipment: CTV
    frequency: 0
failures:
  -
    scale: 0
    shape: 0
    time: 0
    materials: 0
    operation_reduction: 0
    service_equipment: CAB
    level: 1
    description: n/a
```

## Set Up the Simulation

In the remaining sections of the tutorial, we will work towards setting up and running
a simulation.

### Define the data library path

The set library enables WOMBAT to easily access and store data files consistently. Here,
the `DINWOODIE` reference is going to be used again.

```{note}
If a custom library is being used, the `library_path` must be the full path name to the
location of the folder where the configuration data is contained.
```

```{code-cell} ipython3
library_path = DINWOODIE  # or user-defined path for an external data library
```

### The configuration file

In the configuration below, there are a number of data points that will define our
wind farm layout, weather conditions, working hours, customized start and completion
years, project size, financials, and the servicing equipment to be used. Note that there
can be as many or as few of the servicing equipment units as desired.

The purpose of an overarching configuration file is to provide a single place to define
the primary inputs for a simulation. Below the base configuration is loaded and displayed
with comments to show where each of files are located in the library structure. WOMBAT
will know where to go for these pointers when the simulation is initialized so the data
is constructed and validated correctly.

```{note}
As of veraion 0.10, all non-CSV file inputs can be difined in a single configuration
file. Please see [the configuration API details](simulation-api:config) for details.
```

```{code-cell} ipython3
config = load_yaml(library_path / "project/config", "base.yaml")
```

```{code-block} yaml
# Contents of: dinwoodie / config / base.yaml
name: dinwoodie_base
weather: alpha_ventus_weather_2002_2014.csv  # located in: dinwoodie / weather
service_equipment:
# YAML-encoded list, but could also be created in standard Python list notation with
# square brackets: [ctv1.yaml, ctv2.yaml, ..., hlv_requests.yaml]
# All below equipment configurations are located in: dinwoodie / vessels
  - ctv1.yaml
  - ctv2.yaml
  - ctv3.yaml
  - fsv_requests.yaml
  - hlv_requests.yaml
layout: layout.csv  # located in: dinwoodie / windfarm
inflation_rate: 0
fixed_costs: fixed_costs.yaml  # located in: dinwoodie / project / config
workday_start: 7
workday_end: 19
start_year: 2003
end_year: 2012
project_capacity: 240
# port: base_port.yaml  <- When running a tow-to-port simulation the port configuration
#                          pointer is provided here and located in: dinwoodie / project / port
```

```{note}
As of v0.10, multiple vessels can be modeled with a single configuration, and vessel,
turbine, cable, and substation configurations can be included directly in the contents
of the primary configuration.
```

Alternatively, the configuration can be simplified to be of the following form to utilize
repeated vessel configurations and the single file configuration. The new keys for where
to embed the data directly correspond to the folder names where the files originally
resided. For instance, below embedding `<library>/vessels/ctv.yaml` into the
main configuration means that we need a new key called `vessels`, with a subkey
underneath called `ctv`.

```{code-block} yaml
name: dinwoodie_base
weather: alpha_ventus_weather_2002_2014.csv  # located in: dinwoodie / weather
service_equipment:
# YAML-encoded list, but could also be created in standard Python list notation with
# square brackets: [ctv1.yaml, ctv2.yaml, ..., hlv_requests.yaml]
# All below equipment configurations are located in: dinwoodie / vessels
  - [ctv, 3]
  - fsv_requests.yaml
  - hlv_requests.yaml
layout: layout.csv  # located in: dinwoodie / windfarm
inflation_rate: 0
fixed_costs: fixed_costs.yaml  # located in: dinwoodie / project / config
workday_start: 7
workday_end: 19
start_year: 2003
end_year: 2012
project_capacity: 240
vessels:
  ctv:
    ... # contents of ctv1.yaml without the numbered naming which is created automatically
turbines:
  ...
cables:
  ...
substations:
  ...
```

For a complete example of this, please see the
`library/corewind/project/config/morro_bay_in_situ_consolidated.yaml` configuration
file.

## Create a simulation

There are two ways that this could be done, the first is to use the classmethod
`Simulation.from_config()`, which allows for the full path string, a dictionary, or
`Configuration` object to passed as an input, and the second is through a standard
class initialization.

### Option 1: `Simulation.from_config()`

Load the file from the `Configuration` object that was created in the prior code black

```{code-cell} ipython3

sim = Simulation.from_config(library_path=library_path, config=config)

# Delete any files that get initialized through the simulation environment
sim.env.cleanup_log_files()
```

### Option 2: `Simulation()`

Load the configuration file automatically given a library path and configuration file name.

In this usage, the string "DINWOODIE" can be used because the `Simulation` class knows
to look for this library mapping, as well as the "IEA_26" mapping for the two validation
cases that we demonstrate in the examples folder.

```{note}
In Option 2, the config parameter can also be set with a dictionary.

The library path in the configuration file should match the one provided, or the
setup steps will fail in the simulation.
```

```{code-cell} ipython3
sim = Simulation(
    library_path="DINWOODIE",  # automatically directs to the provided library
    config="base.yaml"
)
sim.env.cleanup_log_files()
```

### Seeding the simulation random variable

Using `random_seed` a simulation can be seeded to produce the same results every single
time, or a `random_generator` can be provided to use the same generator for a batch of
identical simulations to better understand the variations in results.

```{code-cell} ipython3
sim = Simulation(
    library_path="DINWOODIE",  # automatically directs to the provided library
    config="base.yaml",
    random_seed=2023,  # integer value indicating how to seed the internally-created generator
)
sim.env.cleanup_log_files()

rng = np.random.default_rng(seed=2023)  # create the generator
sim = Simulation(
    library_path="DINWOODIE",  # automatically directs to the provided library
    config="base.yaml",
    random_generator=rng,  # generator that can be shared among all processes
)
```

## Run the analysis

When the run method is called, the default run time is for the full length of the
simulation, however, if a shorter run than was previously designed is required for
debugging, or something similar, we can use `sum.run(until=<your-time>)` to do this. In
the `run` method, not only is the simulation run, but the metrics class is loaded at the
end to quickly transition to results aggregation without any further code.

```{code-cell} ipython3
# Timing for a demonstration of performance
start = perf_counter()

sim.run()

end = perf_counter()

timing = end - start
print(f"Run time: {timing / 60:,.2f} minutes")
```

## Metric computation

For a more complete view of what metrics can be compiled, please see the [metrics notebook](metrics_demonstration.md), though for the sake of demonstration a few methods will
be shown here

```{code-cell} ipython3
net_cf = sim.metrics.capacity_factor(which="net", frequency="project", by="windfarm").values[0][0]
gross_cf = sim.metrics.capacity_factor(which="gross", frequency="project", by="windfarm").values[0][0]
print(f"  Net Capacity Factor: {net_cf:2.1%}")
print(f"Gross Capacity Factor: {gross_cf:2.1%}")
```

```{code-cell} ipython3
# Report back a subset of the metrics
total = sim.metrics.time_based_availability(frequency="project", by="windfarm")
print(f"  Project time-based availability: {total.windfarm[0]:.1%}")

total = sim.metrics.production_based_availability(frequency="project", by="windfarm")
print(f"Project energy-based availability: {total.windfarm[0]:.1%}")

total = sim.metrics.equipment_costs(frequency="project", by_equipment=False)
print(f"          Project equipment costs: ${total.values[0][0] / sim.metrics.project_capacity:,.2f}/MW")

```

## Optional: Delete the logging files

In the case that a lot of simulations are going to be run, and the processed outputs
are all that is required, then there is a convenience method to clean up these files
automatically once you are done.

```{code-cell} ipython3
sim.env.cleanup_log_files()
```
