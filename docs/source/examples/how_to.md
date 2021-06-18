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
from pprint import pprint  # better print formatiting
from pathlib import Path  # a better path management system

import numpy as np
import pandas as pd

from wombat.core import Simulation, Configuration
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
purposes of this tutorial, we'll be working with the data under `library/dinwoodie` in
the Github repository, and specifically the base case.

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

Please see the `FixedCosts` API documentation for details on this optional piece of
financial modeling.

### Financial Model (SAM)

WOMBAT uses PySAM to model the financials of the windfarm and retrieve higher-level cost
metrics only data is passed into the `SAM_settings` filed in the main configuration. An
input file exists, but is only for demonstration purposes, and does not contain
meaningful inputs.


### The System Models

The higher-level systems (assets) of a windfarm are the cables, turbine(s), and
substation(s), each with their own model, though heavily overlapping. Within each of
these systems, there are modeled, and pre-defined, subassemblies (or componenents) that
rely on two types of maintenance tasks:
 - maintenance: scheduled maintenance tasks
 - failures: unscheduled maintenance tasks

Generally, though in this case for the electrical_system subassembly it is not modeled
and has no inputs, the data inputs look like the below example. For a thorough definition,
it is recommended to see the API documentation of the
[Maintenance](../API/types.md#maintenance-tasks) and
[Failure](../API/types.md#failures) data classes. It should be noted that the yaml
defintion below specifies that maintenance tasks are in a bulleted list format and that
failure defintions require a dictionary-style input with keys to match the severity
level of a given failure.

```{note}
For all non-modeled parameters, inputs must still be provided, though they can be all
zeros. This may become more flexible over time, but for now, at least on maintenance
and one failure must be provided even when all zeros.
```

```
electrical_system:
  name: electrical_system
  maintenance:
  - description: n/a
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
      service_equipment: CTV
      operation_reduction: 0
      level: 1
      description: n/a
```

A more complete example would be for the generator subassembly, as follows:
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

The following is taken from `windfarm/offshore_substation.yaml` and is a good example of
a non-modeled system.

```
capacity_kw: 0
capex_kw: 0
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
Similar to the substation, it relies mainly on the subsystem model with a few extra
parameters, as defined here:

capacity_kw
: The capacity of the system. Only needed if a $/kw cost basis is being used.

capex_kw
: The $/kw cost of the machine, if not providing absolute costs.

power_curve: file
: File that provides the power curve definition.

power_curve: bin_width
: Distince in (m/s) between two points on the power curve.

The `windfarm/vestas_v90.yaml` data file provides the following definition.

```
capacity_kw: 3000
capex_kw: 1300  # need an updated value
power_curve:
  file: vestas_v90_power_curve.csv
  bin_width: 0.5
```

The power curve input CSV contains two columns `windspeed_ms` and `power_kw` that should
be defined using the windspeed for a bin, in m/s and the power produced at that
windspeed, in kW.

In addition to the above, the following subassembly definitions must be provided in
a similar manner to the substation transformer.

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
Currently, only array cables are modeled at this point in time, though in the future
they will be enabled.
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
see the [ServiceEquipment API documentation](../API/types.md#service-equipment)

Below is an definition of the different equipment codes and their designations to show
the breadth of what can be simulated.

RMT
: remote (no actual equipment BUT no special implementation), akin to remote resets

DRN
: drone, or potentially even helicopters by changing the costs

CTV
: crew transfer vessel/vehicle

SCN
: small crane (i.e., field support vessel)

LCN
: large crane (i.e., heavy lift vessel)

CAB
: cabling-specific vessel/vehicle

DSV
: diving support vessel


## Set Up the Simulation

In the remaining sections of the tutorial, we will work towards setting up and running
a simulation.

### Define the data library path

The set library enables WOMBAT to easily access and store data files in a consistent
manner. Here the `DINWOODIE` reference is going to be used again.


```{code-cell} ipython3
library_path = DINWOODIE
```

### Load the configuration file

```{note}
At this stage, the path to the configuration will need to be created manually.
```

In this configuration we've provide a number of data points that will define our windfarm layout, weather conditions, working hours, customized start and completion years, project size, financials, and the servicing equipment to be used. Note that there can be as many or as
few of the servicing equipment as desired.

```{code-cell} ipython3
config = load_yaml(str(library_path / "config"), "base.yaml")

for k, v in config.items():
  print(f"\033[1m{k}\033[0m:\n  {v}")  # make the keys bold
```

## Instantiate the simulation

There are two ways that this could be done, the first is to use the classmethod `Simulation.from_inputs()`, and the second is through a standard class initialization.

```{code-cell} ipython3
# Option 1
sim = Simulation.from_inputs(**config)

# Delete the .log files that get initialized
sim.env.cleanup_log_files(log_only=True)

# Option 2
# Note here that a string "DINWOODIE" is passed because the Simulation class knows to
# retrieve the appropriate path
simulation_name = "example_dinwoodie"
sim = Simulation(
    name=simulation_name,
    library_path="DINWOODIE",
    config="base.yaml"
)
```
```{note}
In Option 2, the config parameter can also be set with a dictionary.
```


## Run the analysis

When the run method is called, the default run time is for the full length of the simulation, however, if a shorter run than was previously designed is required for debugging, or something similar, we can use `sum.run(until=<your-time>)` to do this. In the `run` method, not
only is the simulation run, but the metrics class is loaded at the end to quickly transition
to results aggregation without any further code.

```{warning}
It should be noted at this stage that a run time that isn't divisible by 8760 (hours in a year), the run will fail at the end due to the `Metrics` class's reliance on PySAM at the initialization stage. This will be worked out in later iterations.

Users should also be careful of leap years because the PySAM model cannot handle them,
though if Feb 29 is provided, it will be a part of the analysis and stripped out before
being fed to PySAM.
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
