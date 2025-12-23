# Examples Reference

This page will provide a brief overview of the varying examples available in the
[examples folder on GitHub](https://github.com/WISDEM/WOMBAT/blob/main/examples).

## `archival/`

This contains the old notebooks and data used to prepare the unveiling of the model to
DOE. The results are displayed in
[presentations section](presentations:fy20-doe) of the documentation.

## Ongoing Analysis Work & Default Data Results Demonstration

### `COWER_om_worflow.ipynb`

This workflow example demonstrates the new (as of v0.13) default offshore wind data sets and the
O&M portion of the NREL Cost of Wind Energy Review (COWER) analysis. The underlying data and example
are kept up to date every year for this project, so the data will be updated annually (as of 2025).

## Explanations

The following notebooks are aimed at demonstrating and explaining various functionality
to users.

### `how_to.ipynb`

This is a Jupyter Notebook version of the [*How To* section](./how_to.md) of the
documentation that allows users to interact with the walk through.

### `strategy_demonstration.ipynb`

This is a Jupyter Notebook version of the
[*Strategy Demonstration* section](./strategy_demonstration.md) of the documentation
that allows users to interact with the walk through.

### `default_data_demonstration.ipynb`

This is a Jupyter Notebook version of the
[*Default Data Demonstration* section](./default_data_demonstration.md) of the documentation
that allows users to interact with the examples.

### `metrics_demonstration.ipynb`

This is a Jupyter Notebook version of the
[*Metrics Demonstration* section](./metrics_demonstration.md) of the documentation that
allows users to interact with the varying maintenance strategy models.

### `NAWEA_interactive_walkthrough.ipynb`

This is the notebook used for the NAWEA WindTech 2023 Workshop. See
[here](../workshops/nawea_wind_tech_2023.md) for details.

### `electrolyzer_example.ipynb`

Demonstrates how to set up and run a standalone electrolyzer simulation within WOMBAT's
wind farm framework.

## Validation

```{note}
These example are no longer recalibrated for updated releases, and are merely kept up
to date with WOMBAT's input changes.
```

### `avanessova_2025.ipynb`

This example demonstrates the WOMBAT analysis portion of the 2025 doctoral dissertation by
Nadezda Avanessova [^avanessova2025], which in part focuses on a comparison of
ORE Catapult's COMPASS, WavEC's O&M Tool, and NREL's WOMBAT for offshore wind O&M. This example
was originally run in v0.8.1, and updated to v0.12, so some results will differ from what was published.

### `dinwoodie_validation.ipynb`

This shows the latest results of the model validation using the modeling parameters from
Dinwoodie, et al., 2015 [^dinwoodie2015reference].

### `iea_26_validation.ipynb`

This shows the latest results of the model validation using the modeling parameters from
IEA, 2016 [^smart2016iea].

### `timing_benchmarks.ipynb`

This notebook is a simple benchmark for comparing run times between releases from NREL's
FY22 & FY23. That said, there has been a change in computers since running this, and
the listed times for v0.7 in the notebooks will not match those in this example.

[^dinwoodie2015reference]: Iain Dinwoodie, Ole-Erik V Endrerud, Matthias Hofmann, Rebecca Martin, and Iver Bakken Sperstad. Reference cases for verification of operation and maintenance simulation models for offshore wind farms. *Wind Engineering*, 39(1):1–14, 2015.
[^smart2016iea]: Gavin Smart, Aaron Smith, Ethan Warner, Iver Bakken Sperstad, Bob Prinsen, and Roberto Lacal-Arantegui. Iea wind task 26: offshore wind farm baseline documentation. Technical Report, National Renewable Energy Lab.(NREL), Golden, CO (United States), 2016.
[^avanessova2025]: Avanessova, Nadezda. "Developing a holistic operation and maintenance simulation tool for emerging offshore wind projects." (2025).
