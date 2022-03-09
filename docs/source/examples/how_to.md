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

This tutorial will walk through the setup, running, and results stages of a WOMBAT simulation.

## Imports

The following code block demonstrates a typical setup for working with WOMBAT and running
analyses.

```{code-cell} ipython3
from time import perf_counter  # timing purposes only
from pprint import pprint

import numpy as np
import pandas as pd

from wombat.core import Simulation
from wombat.core.library import load_yaml, DINWOODIE

# Seed the random variable for consistently randomized results
np.random.seed(0)

# Improve ability to read longer outputs
pd.set_option("display.float_format", '{:,.2f}'.format)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
```


## Defining the Simulation

The following will demonstrate the required information to run a simulation. For the
purposes of this tutorial, we'll be working with the data under
`library/code_comparison/dinwoodie` in the Github repository, and specifically the base
case.

````{note}
One important item to note is that the library structure is enforced within the code so all
data must be placed in the appropriate locations in your analysis' library as follows:
```
<libray path>
├── config                <- Simulation configuration files
├── windfarm              <- Windfarm layout file(s); turbine, substation, and cable configurations
├── outputs
│   ├── logs              <- The raw anaylsis log files
│   ├── metrics           <- A place to save metrics/computed outputs.
│   ├── <self-defined>    <- Any other folder you choose for saving outputs (not enforced)
├── repair
│   ├── transport         <- Servicing equipment configurations
├── weather               <- Weather profiles
```

As a convenience feature you can import the provided validation data libraries as
`DINWOODIE` or `IEA_26` as is seen in the imports above, and a consistent path will be
enabled.
````


### Windfarm Layout

The windfarm layout is determined by a csv file, `dinwoodie/windfarm/layout.csv` in this case. Below is a sample of what information is required and how to use each field, followed by a visual example. It should be noted that none of the headings are case sensitive.


id (required)
: Unique identifier for the asset; no spaces allowed.

substation_id (required)
: The id field for the substation that the asset connects to; in the case that this is a substation, then this field should be the same as id; no spaces allowed.

name (required)
: A descriptive name for the turbine, if desired. This can be the same as id.

longitude (optional)
: The longitudinal position of the asset, can be in any geospatial reference; optional.

latitude (optional)
: The latitude position of the asset, can be in any geospatial reference; optional.

string (required)
: The integer, zero-indexed, string number for where the turbine will be positioned.

order (required)
: The integer, zero-indexed position on the string for where the turbine will be positioned.

distance (optional)
: The distance to the upstream asset; if this is calculated (input = 0), then the straightline distance is calculated using the provided coordinates (WGS-84 assumed).

subassembly (required)
: The file that defines the asset's modeling parameters.

upstream_cable (required)
: The file that defines the upstream cable's modeling parameters.

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

| id | substation_id | name | longitude | latitude | string | order | distance | subassembly | upstream_cable |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | :-- | :-- |
| OSS1 | OSS1 | OSS1 | 0 | 0 |  |  |  | offshore_substation.yaml | export.yaml |
| S00T1 | OSS1 | S00T1 | 0 | 0 | 0 | 0 | 0 | vestas_v90.yaml | array.yaml |
| S00T2 | OSS1 | S00T2 | 0 | 0 | 0 | 1 | 0 | vestas_v90.yaml | array.yaml |
| S00T3 | OSS1 | S00T3 | 0 | 0 | 0 | 2 | 0 | vestas_v90.yaml | array.yaml |
| ... |
| S00T79 | OSS1 | S00T79 | 0 | 0 | 0 | 78 | 0 | vestas_v90.yaml | array.yaml |
| S00T80 | OSS1 | S00T80 | 0 | 0 | 0 | 79 | 0 | vestas_v90.yaml | array.yaml |


### Weather Profile

The weather profile will broadly define the simulation range with its start and stop
points, though a narrower one can be used when defining the simulation (more later).

The following columns should exist in the data set with the provided guidelines.
datetime (required)
: A date and time stamp, any format.

windspeed (required)
: Unique identifier for the asset; no spaces allowed.

waveheight (optional)
: The hourly, mean waveheight, in meters, for the time stamp. If waves are not required,
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


### Fixed Costs

Please see the [`FixedCosts` API documentation](../API/types.md#fixed-cost-model) for
details on this optional piece of financial modeling.


### Financial Model (SAM)

WOMBAT uses PySAM to model the financials of the windfarm and retrieve higher-level cost
metrics only data is passed into the `SAM_settings` filed in the main configuration. An
input file exists, but is only for demonstration purposes, and does not contain
meaningful inputs.


### The System Models

The actual assets on the windfarm such as cables, turbines, and substations, are
referred to as systems in WOMBAT, and each has their own individual model. Within each
of these systems, there are modeled, and pre-defined, subassemblies (or componenents) that
rely on two types of repair models:
 - maintenance: scheduled, fixed time interval-based maintenance tasks
 - failures: unscheduled, Weibull distribution-based modeled, maintenance tasks

In the example below we show a `generator` subassembly with an annual service task and
frequent manual reset. For a thorough definition, it is recommended to read the API
documentation of the [Maintenance](../API/types.md#maintenance-tasks) and
[Failure](../API/types.md#failures) data classes. It should be noted that the yaml
defintion below specifies that maintenance tasks are in a bulleted list format and that
failure defintions require a dictionary-style input with keys to match the severity
level of a given failure. For more details on the complete subassembly definition,
please visit the [Subassembly API documentation](../API/types.md#subassembly-model).

```
generator:
  name: generator
  maintenance:
  - description: annual service
    time: 60
    materials: 18500
    service_equipment: CTV
    frequency: 365
  failures:
    1:
      scale: 0.1333
      shape: 1
      time: 3
      materials: 0
      service_equipment: CTV
      operation_reduction: 0.0
      level: 1
      description: manual reset
```


#### Substations

The substation model relies on two specific inputs, and one subassembly input (transformer).

capacity_kw
: The capacity of the system. Only needed if a $/kw cost basis is being used.

capex_kw
: The $/kw cost of the machine, if not providing absolute costs.

transformer
: See the subassembly model for more details.

The following is an example of substation yaml definition with no modeled subasemblies.

```
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
    1:
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

The turbine has the most to define out of the three systems in the windfarm model.
Similar to the substation, it relies mainly on the subassembly model with a few extra
parameters, as defined here:

capacity_kw
: The capacity of the system. Only needed if a $/kw cost basis is being used.

capex_kw
: The $/kw cost of the machine, if not providing absolute costs.

power_curve: file
: File that provides the power curve definition.

power_curve: bin_width
: Distince in (m/s) between two points on the power curve.

The `windfarm/vestas_v90.yaml` data file provides the following definition in addition
to the the maintenance and failure definitions that were shown previously.

```
capacity_kw: 3000
capex_kw: 1300
power_curve:
  file: vestas_v90_power_curve.csv
  bin_width: 0.5
```

The power curve input CSV requires the following two columns: `windspeed_ms` and
`power_kw` that should be defined using the windspeed for a bin, in m/s and the power produced at that
windspeed, in kW. The current method available for generating the power curve is the IEC
61400-12-1-2 method for a wind-speed binned power curve. If there is a need/desire for
additional power curve methodologies, then [please submit an issue on the GitHub](https://github.com/WISDEM/WOMBAT/issues)!

For an open source listing of a variety of land-based, offshore, and distributed wind
turbine power curves, please visit the
[NREL Turbine Models repository](https://github.com/NREL/turbine-models).

In addition to the above, the following subassembly definitions can all be modeled,
though, and only one is required for the model to successfully be created.

 - electrical_system
 - electronic_control
 - sensors
 - hydraulic_system
 - yaw_system
 - rotor_blades
 - mechanical_brake
 - rotor_hub
 - gearbox
 - generator
 - supporting_structure
 - drive_train


#### Cables

```{note}
Currently, only array cables are modeled, though in the future export cables will be enabled.
```

The array cable is the simplest format in that you only define a descriptive name,
and the maintenance and failure events as below.

```
name: array cable
maintenance:
  -
    description: n/a
    time: 0
    materials: 0
    service_equipment: CTV
    frequency: 0
failures:
  1:
    scale: 0
    shape: 0
    time: 0
    materials: 0
    operation_reduction: 0
    service_equipment: CAB
    level: 1
    description: n/a
```

## Servicing Equipment

The servicing equipment currently run on an scheduled visit basis for the sake of
simplicity in this early stage of the model. This may seem to be a limitation, there are
plenty of workarounds in this approach, by design! For instance, due to the simplicity
of the input model, we can simulate multiple visits throughout the year for large cranes
(e.g., heavy lift vessels, crew transfer vessels, or crawler cranes) enabling similar
behaviors to that of an unscheduled maintenance model.

For complete documentation of how the servicing equipment parameters are defined, please
see the [ServiceEquipmentData API documentation](../API/types.md#service-equipment)

Below is an definition of the different equipment codes and their designations to show
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

LCN
: large crane (i.e., heavy lift vessel or crawler crane)

CAB
: cabling-specific vessel/vehicle

DSV
: diving support vessel

In addition to a variety of servicing equipment types, there is support for
3 different equipment-level dispatch strategies, as described below. For a set of
example scenarios, please see the [strategy demonstration](strategy_demonstration.ipynb).

scheduled
: dispatch servicing equipment for a specified date range each year

requests
: dispatch the servicing equipment once a `strategy_threshold` number of requests
  that the equipment can service has been reached

downtime
: dispatch the servicing equipment once the windfarm's operating level reaches the
  `strategy_threshold` percent operating has been reached.


## Set Up the Simulation

In the remaining sections of the tutorial, we will work towards setting up and running
a simulation.

### Define the data library path

The set library enables WOMBAT to easily access and store data files in a consistent
manner. Here the `DINWOODIE` reference is going to be used again.

```{note}
If a custom library is being used, the `library_path` must be the full path name to the
location of the folder where the configuration data is contained.
```

```{code-cell} ipython3
library_path = DINWOODIE  # or user-defined path for an external data library
```

### The configuration file


In the below configuration we've provided a number of data points that will define our
windfarm layout, weather conditions, working hours, customized start and completion
years, project size, financials, and the servicing equipment to be used. Note that there
can be as many or as few of the servicing equipment as desired.

The purpose of an overarching configuration file is to provide a single place to define
the primary inputs for a simulation. As is seen below most of the inputs are pointers
to other files that WOMBAT will then use to construct and validate the remaining
simulation settings.

```{code-cell} ipython3
config = load_yaml(library_path / "config", "base.yaml")

for k, v in config.items():
  print(f"\033[1m{k}\033[0m:", end="\n  ")  # make the keys bold
  pprint(v, indent=2)
```

## Instantiate the simulation

There are two ways that this could be done, the first is to use the classmethod
`Simulation.from_config()`, which allows for the full path string, a dictionary, or
`Configuration` object to passed as an input, and the second is through a standard
class initialization.

### Option 1: `Simulation.from_config()`

Load the file from the `Configuration` object that was created in the prior code black

```{code-cell} ipython3

sim = Simulation.from_config(config)

# Delete any files that get initialized through the simulation environment
sim.env.cleanup_log_files(log_only=False)
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
```


## Run the analysis

When the run method is called, the default run time is for the full length of the simulation, however, if a shorter run than was previously designed is required for debugging, or something similar, we can use `sum.run(until=<your-time>)` to do this. In the `run` method, not
only is the simulation run, but the metrics class is loaded at the end to quickly transition
to results aggregation without any further code.

```{warning}
It should be noted at this stage that a run time that isn't divisible by 8760 (hours in
a year), the run will fail if the SAM financial model is being used due to a mismatch
with PySAM's requirements and the model's outputs. This will be worked
out in later iterations to cap it to the correct number of hours (future feature) for an
evenly divisible year.

Users should also be careful of leap years because the PySAM model cannot handle them,
though if Feb 29 is provided, it will be a part of the analysis and stripped out before
being fed to PySAM, so no errors will occur.
```

```{code-cell} ipython3
# Timing for a demonstration of performance
start = perf_counter()

sim.run()

end = perf_counter()

timing = end - start
if timing > 60 * 2:
    print(f"Run time: {timing / 60:,.2f} minutes")
else:
    print(f"Run time: {timing:,.4f} seconds")
```


## Metric computation

For a more complete view of what metrics can be compiled, please see the [metrics notebook](metrics_demonstration.ipynb), though for the sake of demonstration a few methods will
be shown here

```{code-cell} ipython3
net_cf = sim.metrics.capacity_factor(which="net", frequency="project", by="windfarm")
gross_cf = sim.metrics.capacity_factor(which="gross", frequency="project", by="windfarm")
print(f"  Net Capacity Factor: {net_cf:2.1f}%")
print(f"Gross Capacity Factor: {gross_cf:2.1f}%")
```

```{code-cell} ipython3
# Report back a subset of the metrics
total = sim.metrics.time_based_availability(frequency="project", by="windfarm")
print(f"  Project time-based availability: {total * 100:.1f}%")

total = sim.metrics.production_based_availability(frequency="project", by="windfarm")
print(f"Project energy-based availability: {total * 100:.1f}%")

total = sim.metrics.equipment_costs(frequency="project", by_equipment=False)
print(f"          Project equipment costs: ${total / sim.metrics.project_capacity:,.2f}/MW")

```


## Optional: Delete the logging files

In the case that a lot of simulations are going to be run, and the processed outputs are all that is required, then there is a convenience method to cleanup these files automatically once you are done.

```{code-cell} ipython3
sim.env.cleanup_log_files(log_only=False)
```
