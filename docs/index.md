# WOMBAT - Windfarm Operations and Maintenance cost-Benefit Analysis Tool

[![DOI](https://img.shields.io/badge/DOI-10.2172%2F1894867-brightgreen?link=https://doi.org/10.2172/1894867)](https://www.osti.gov/biblio/1894867)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://badge.fury.io/py/wombat.svg)](https://badge.fury.io/py/wombat)
[![PyPI downloads](https://img.shields.io/pypi/dm/wombat?link=https%3A%2F%2Fpypi.org%2Fproject%2FWOMBAT%2F)](https://pypi.org/project/WOMBAT/)

[![codecov](https://codecov.io/gh/WISDEM/WOMBAT/branch/main/graph/badge.svg?token=SK9M10BZXY)](https://codecov.io/gh/WISDEM/WOMBAT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)
[![Documentation site](https://jupyterbook.org/badge.svg)](https://wisdem.github.io/WOMBAT)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black formatter](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort import formatter](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Overview

The WOMBAT framework is designed to provide an open source tool adhering to FLOSS principles
for the wind farm lifecycle research community. Specifically, WOMBAT is meant to serve as
a what-if, or scenario-based, simulation tool, so that you can model the trade-offs in
decision-making for the operations and maintenance phase of a wind farm.

As a supplement to this documentation site, there is also an NREL Technical Report that
goes through much of the design and implementation details available at:
https://www.osti.gov/biblio/1894867. If you use this software, please cite it using the
following BibTeX information, or in commonly used citation formats
[here](https://www.osti.gov/biblio/1894867).

```bibtex

   @techreport{hammond2022wombat,
      title = {Windfarm Operations and Maintenance cost-Benefit Analysis Tool (WOMBAT)},
      author = {Hammond, Rob and Cooperman, Aubryn},
      doi = {10.2172/1894867},
      url = {https://www.osti.gov/biblio/1894867},
      place = {United States},
      year = {2022},
      month = {10},
      institution = {National Renewable Energy Lab. (NREL)},
   }
```

For any questions, feel free to open up an issue in the repository or email:
rob.hammond@nrel.gov.

## Latest Changes?

As of v0.10, a series of convenience features and consistency updates have been made.
- A breaking change to failure data has been made by using lists instead of dictionaries
  in the configuration of cables, turbines, and substations. To ease adoption, a
  function has been provided to convert to the new format:
  https://wisdem.github.io/WOMBAT/API/utilities.html#importing-and-converting-from-old-versions.
- Single configuration files are now supported for all non-CSV data. This means that
  servicing equipment, cables, turbines, and substations can all be included in the
  primary configuration file.
- Simplification of duplicated servicing equipment definitions: simply provide a list
  of the filename/id and the number of them that should be in the simulation.
- Date-based maintenance scheduling is now possible.

Please see the CHANGELOG for complete information, and the appropriate documentation
pages.

* On this site: https://wisdem.github.io/WOMBAT/changelog.html
* On GitHub: https://github.com/WISDEM/WOMBAT/blob/main/CHANGELOG.md

## The Model in 30 Seconds Or Less

In general, the model has 2 overarching branches: the wind farm itself (the technology
strategy), and the simulation environment (the maintenance strategy). For the wind farm
model we can control the varying assets (system in the code)--substations, turbines, and
cables--as well as the components that comprise each asset (subassemblies in the code).
This separation allows for each turbine, cable, or substation component to have its own
unique failure and maintenance models.

As for the environment, this is where the discrete event simulation itself happens, in
addition to logging, repair logic, and other necessary modeling pieces. The image
below provides a more visual representation of this description.

### High Level Architecture

The code is largely broken up into two categories: the wind farm and objects contained
within it, and the simulation and simulation environment components. The wind farm is
composed of systems: substation(s), cables, and turbines, and each of those systems is
composed of subassemblies (a conglomerate of components). For the simulation environment,
we consider all the pieces that allow the simulation to happen such as the API,
servicing equipment, repair manager to hold and pass on tasks, and results post-processing.

```{image} images/high_level_diagram.svg
```

### Simulation Architecture

In the diagram below, we demonstrate the lifecycle of the simulation through the
lifecycle of a single failure.

1) The maximum length of the simulation is defined by the amount of wind and wave
   time series data is provided for the simulation.
2) Each subassembly failure model is a random sampling from a Weibull distribution, so
   for the sake of clarity we'll consider this to be a catastrophic drivetrain failure.
   When the timeout (time to failure) is reached in the simulation, the subassembly's operating level is
   reduced to 0%, and a message is passed to the turbine level (the overarching system
   model).
3) From there, the turbine will shut off, and signal to all other subassembly models to
   not accrue time for their respective maintenance and failure timeouts. Then, the
   turbine creates a repair request and passes that to the repair manager.
4) The repair manager will store the submitted task and depending on the servicing
   equipment's maintenance strategy, a crane or vessel will be called to site.
5) The vessel (or crane) will mobilize and accrue time there, all the while, the turbine
   and wind power plant will accrue downtime. When the servicing equipment arrives at
   the site it will operate according to its operation limits, current weather
   conditions, and site/equipment-specific working hours and continue to log costs and
   downtime.
6) When the servicing is complete, the subassembly will be placed back to good-as-new
   condition and the turbine will be reset to operating. From there all the turbine's
   and drivetrain's failure and maintenance models be turned back on, and the simulation
   will continue on in the same manner until it reaches its user- or weather-defined
   ending point.

```{image} images/simulation_diagram.svg
```

## License

Notice on the NREL application of the Apache-2 license, also found on the
[GitHub](https://github.com/WISDEM/WOMBAT/blob/main/NOTICE), along with the
complete [license](https://github.com/WISDEM/WOMBAT/blob/main/LICENSE) details.

```{include} ../NOTICE
```
