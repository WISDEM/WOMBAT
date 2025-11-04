# Default Library

## Overview
The **default library** provides a consistent set of cost and performance inputs for offshore wind energy systems, standardized to **2024 USD**. This library supports reproducible analyses in the *Cost of Wind Energy Review (COWER): 2025 Edition*, including both fixed-bottom and floating offshore wind cases.

Unlike other datasets in this repository, which largely reflect publicly available sources, this dataset incorporates **internal adjustments and harmonizations** to align with research-focused scenarios.

## Core Assumptions

- **Cost Year:** All monetary values are standardized to **2024 USD**.
- **Data Integration:** Inputs were consolidated from multiple public sources and internal records (`WOMBAT_cost_history.xlsx`), with historical exchange rates and inflation applied to transform costs to 2024 USD.
- **Technology Coverage:** Includes representative inputs for offshore (fixed-bottom and floating) technologies.

## Intended Use

- Serve as a **baseline input** for replicable analyses in the *COWER-2025* results.
- Support **scenario development** and **sensitivity analyses** exploring the impact of cost evolution, operational performance, and logistics assumptions.
- Provide a **controlled reference** for comparison against other datasets.

## Reproducibility

The accompanying notebook, **`examples/COWER-2025-prelim-results.ipynb`**, demonstrates how to replicate O&M results from the *Cost of Wind Energy Review: 2025 Edition*. It runs 50 simulations per case and summarizes mean and standard deviation results to identify sources of variability within cost components.

## Notes

- `WOMBAT_cost_history.xlsx` is an internal NREL document. For questions, contact **Daniel.MulasHernando@nrel.gov**.
- This dataset should be treated as a **scenario-based reference**, not as purely empirical or historical data.
