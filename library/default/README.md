# Default Library

## Overview
The **default library** provides a consistent set of cost and performance inputs for offshore wind
energy systems, standardized to **2024 USD**. This library supports reproducible analyses in the
*Cost of Wind Energy Review (COWER): 2025 Edition*, including both fixed-bottom and floating
offshore wind cases.

Unlike other datasets in this repository, which largely reflect publicly available sources, this
dataset incorporates **internal adjustments and harmonizations** to align with research-focused
scenarios for NREL products that may require outputs to be in 2024 USD, like for example, the Annual
Technology Baseline, or the Cost of Wind Energy Review.

## Core Assumptions

Material costs for repairs and replacements, as well as failure data, are sourced from COREWIND (2020, 2021). Fixed cost data is primarily derived from BVG Associates (2025a, 2025b), while vessel day rate and mobilization assumptions are compiled from a range of public sources. For detailed reference information, please contact Daniel Mulas Hernando (Daniel.MulasHernando@nrel.gov) to request access to WOMBAT_cost_history.xlsx.

- **Cost Year:** All monetary values are standardized to **2024 USD**.
- **Data Integration:** Inputs were consolidated from multiple public sources and internal records (`WOMBAT_cost_history.xlsx`), with historical exchange rates and inflation applied to transform costs to 2024 USD.
- **Technology Coverage:** Includes representative inputs for offshore (fixed-bottom and floating) technologies.

## Intended Use

- Serve as a **baseline input** for replicable analyses in the *COWER-2025* results.
- Support **scenario development** and **sensitivity analyses** exploring the impact of cost evolution, operational performance, and logistics assumptions.

## Reproducibility

The accompanying notebook, **`examples/COWER_om_workflow.ipynb`**, demonstrates how to replicate O&M results from the *Cost of Wind Energy Review: 2025 Edition*. It runs 50 simulations per case and summarizes mean and standard deviation results to identify sources of variability within cost components.

## Notes

- `WOMBAT_cost_history.xlsx` is an internal NREL document. For questions, contact **Daniel.MulasHernando@nrel.gov**.
- This dataset should be treated as a **scenario-based reference**, not as purely empirical or historical data.

## Data Sources

- COREWIND, 2020: https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D6.1-General-frame-of-the-analysis-and-description-of-the-new-FOW-assessment-app.pdf
- COREWIND, 2021: https://corewind.eu/wp-content/uploads/files/publications/COREWIND-D4.2-Floating-Wind-O-and-M-Strategies-Assessment.pdf
- BVG Associates, 2025a: https://guidetoanoffshorewindfarm.com/wind-farm-costs/
- BVG Associates, 2025b: https://guidetofloatingoffshorewind.com/wind-farm-costs/
