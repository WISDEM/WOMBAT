# Default Library

The default library provides a validated, ready-to-use data set for fixed and floating offshore wind,
and an experimental land-based data set. For all three, users can use the pre-configured base models
or use them as a starting point for building custom models.

## Overview

### Offshore

The **default library** provides a consistent set of cost and performance inputs for offshore wind
energy systems, standardized to 2024 USD. This library supports reproducible analyses in the
*Cost of Wind Energy Review (COWER): 2025 Edition*, including both fixed-bottom and floating
offshore wind cases.

Unlike other datasets in this repository, which largely reflect publicly available sources, this
dataset incorporates **internal adjustments and harmonizations** to align with research-focused
scenarios for NREL products that may require outputs to be in 2024 USD, like for example, the Annual
Technology Baseline, or the Cost of Wind Energy Review.

### Land-Based

The default library provides an experimental set of land-based data that is hybridized from a series
of incomplete sources of offshore and onshore costs, and scaling factors between them. It is highly
recommended to merely use this data as a starting point for any analysis work.

## Core Assumptions

### Offshore

Material costs for repairs and replacements, as well as failure data, are sourced from COREWIND (2020, 2021). Fixed cost data is primarily derived from BVG Associates (2025a, 2025b), while vessel day rate and mobilization assumptions are compiled from a range of public sources. For detailed reference information, please contact Daniel Mulas Hernando (Daniel.MulasHernando@nrel.gov) to request access to WOMBAT_cost_history.xlsx.

- **Cost Year:** All monetary values are standardized to **2024 USD**.
- **Data Integration:** Inputs were consolidated from multiple public sources and internal records (`WOMBAT_cost_history.xlsx`), with historical exchange rates and inflation applied to transform costs to 2024 USD.
- **Technology Coverage:** Includes representative inputs for offshore (fixed-bottom and floating) technologies.

### Land-Based

Failure rates, repair times, and materials costs not provided in COREWIND (2020) are are largely
taken from Carroll, et al. (2016) with some custom substitutions or modifications. All failure rates
are converted to a Weibull scale factor. Using Carroll, et al. (2016) we adjust offshore Weibull
scale factors for minor repairs for the generator, blades, pitch system, yaw system, and drive train
(wind-affected subassemblies) by a factor of 5.912, for non-wind-affected subassemblies a factor of
1.217 is applied, and for all major repairs and replacements, a factor of 2.432 is applied. These
scaling factors are primarily derived from Carroll et, al. (2016).

All costs are rescaled from the 12 MW baseline to a 3.5 MW baseline using the COWER 2025 turbine
CapEx figures of \$1,117/kw-yr for onshore and \$1,770/kw-yr for offshore, a 0.6311 scaling factor.

Additional subassemblies are derived from Carroll, et al. (2016) and scaled using the above
assumptions. Fixed costs are derived from Wiser, et al. (2019), and equipment costs come from
LandBOSSE and internal source adjustments.

## Intended Use

- Serve as a **baseline input** for replicable analyses of the offshore fixed and floating *COWER-2025* results.
- Serve as a **baseline input** for the development of land-based analyses.
- Support **scenario development** and **sensitivity analyses** exploring the impact of cost evolution, operational performance, and logistics assumptions.

## Reproducibility

The accompanying notebook, **`examples/COWER_om_workflow.ipynb`**, demonstrates how to replicate O&M results from the *Cost of Wind Energy Review: 2025 Edition*. It runs 50 simulations per case and summarizes mean and standard deviation results to identify sources of variability within cost components.

## Notes

- `WOMBAT_cost_history.xlsx` is an internal NREL document. For questions, contact **Daniel.MulasHernando@nrel.gov**.
- This dataset should be treated as a **scenario-based reference**, not as purely empirical or historical data.

## Data Sources

[1] COREWIND, 2020: https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D6.1-General-frame-of-the-analysis-and-description-of-the-new-FOW-assessment-app.pdf

[2] COREWIND, 2021: https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D4.2-Floating-Wind-O-and-M-Strategies-Assessment.pdf

[3] BVG Associates, 2025a: https://guidetoanoffshorewindfarm.com/wind-farm-costs/

[4] BVG Associates, 2025b: https://guidetofloatingoffshorewind.com/wind-farm-costs/

[5] Carroll, James, Alasdair McDonald, and David McMillan. "Failure rate, repair time and unscheduled O&M cost analysis of offshore wind turbines." Wind energy 19, no. 6 (2016): 1107-1119.

[6] Wiser, Ryan, Mark Bolinger, and Eric Lantz. "Assessing wind power operating costs in the United States: Results from a survey of wind industry experts." Renewable Energy Focus 30 (2019): 46-57.

[7] National Renewable Energy Laboratory (NREL), "Clean Energy Employment Impacts," (2023). https://www.osti.gov/biblio/1995015.
