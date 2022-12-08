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

# Demonstration of the Available Metrics

For a complete list of metrics and their documentation, please see the API Metrics [documentation](../API/simulation_api.md#metrics-computation).

This demonstration will rely on the results produced in the "How To" notebook and serves as an extension of the API documentation to show what the results will look like depending on what inputs are provided.


```{code-cell} ipython3
from pprint import pprint

import pandas as pd

from wombat.core import Simulation, Metrics

# Clean up the aesthetics for the pandas outputs
pd.set_option("display.float_format", '{:,.2f}'.format)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 10)
```

## Table of Contents

Below is a list of top-level sections to demonstrate how to use WOMBAT's `Metrics` class methods and an explanation of each individual metric.
 - [Setup](#setup): Running a simulation to gather the results
 - [Availability](#availability): Time-based and energy-based availability
 - [Capacity Factor](#capacity-factor): Gross and net capacity factor
 - [Task Completion Rate](#task-completion-rate): Task completion metrics
 - [Equipment Costs](#equipment-costs): Cost breakdowns by servicing equipment
 - [Service Equipment Utilization Rate](#service-equipment-utilization-rate): Utilization of servicing equipment
 - [Vessel-Crew Hours at Sea](#vessel-crew-hours-at-sea): Number of crew or vessel hours spent at sea
 - [Number of Tows](#number-of-tows): Number of tows breakdowns
 - [Labor Costs](#labor-costs): Breakdown of labor costs
 - [Equipment and Labor Costs](#equipment-and-labor-costs): Combined servicing equipment and labor cost breakdown
 - [Component Costs](#component-costs): Materials costs
 - [Fixed Cost Impacts](#fixed-cost-impacts): Total fixed costs
 - [OpEx](#opex): Project OpEx
 - [Process Times](#process-times): Timing of various stages of repair and maintenance
 - [Power Production](#power-production): Potential and actually produced power
 - [Net Present Value](#net-present-value): Project NPV calculator
 - [PySAM-Powered Results](#pysam-powered-results): PySAM results, if configuration is provided


(setup)=
## Setup

The simulations from the How To notebook are going to be rerun as it is not recommended to create a Metrics class from scratch due to the
large number of inputs that are required and the initialization is provided in the simulation API's run method.

To simplify this process, a feature has been added to save the simulation outputs required to generate the Metrics inputs and a method to reload those outputs as inputs.
