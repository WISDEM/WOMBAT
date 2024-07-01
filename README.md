# WOMBAT: Windfarm Operations & Maintenance cost-Benefit Analysis Tool

[![DOI 10.2172/1894867](https://img.shields.io/badge/DOI-10.2172%2F1894867-brightgreen?link=https://doi.org/10.2172/1894867)](https://www.osti.gov/biblio/1894867)
[![PyPI version](https://badge.fury.io/py/wombat.svg)](https://badge.fury.io/py/wombat)
[![codecov](https://codecov.io/gh/WISDEM/WOMBAT/branch/main/graph/badge.svg?token=SK9M10BZXY)](https://codecov.io/gh/WISDEM/WOMBAT)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples)
[![Jupyter Book](https://jupyterbook.org/badge.svg)](https://wisdem.github.io/WOMBAT)

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

This library provides a tool to simulate the operation and maintenance phase (O&M) of
distributed, land-based, and offshore windfarms using a discrete event simultaion
framework.

WOMBAT is written around the [`SimPy`](https://gitlab.com/team-simpy/simpy) discrete
event simulation framework. Additionally, this is supported using a flexible and modular
object-oriented code base, which enables the modeling of arbitrarily large (or small)
windfarms with as many or as few failure and maintenance tasks that can be encoded.

Please note that this is still heavily under development, so you may find some functionality
to be incomplete at the current moment, but rest assured the functionality is expanding.
With that said, it would be greatly appreciated for issues or PRs to be submitted for
any improvements at all, from fixing typos (guaranteed to be a few) to features to
testing.

If you use this library please cite our NREL Technical Report:

```bibtex
   @techreport{hammond2022wombat,
      title = {Windfarm Operations and Maintenance cost-Benefit Analysis Tool (WOMBAT)},
      author = {Hammond, Rob and Cooperman, Aubryn},
      abstractNote = {This report provides technical documentation and background on the newly-developed Wind Operations and Maintenance cost-Benefit Analysis Tool (WOMBAT) software. WOMBAT is an open-source model that can be used to obtain cost estimates for operations and maintenance of land-based or offshore wind power plants. The software was designed to be flexible and modular to allow for implementation of new strategies and technological innovations for wind plant maintenance. WOMBAT uses a process-based simulation approach to model day-to-day operations, repairs, and weather conditions. High-level outputs from WOMBAT, including time-based availability and annual operating costs, are found to agree with published results from other models.},
      doi = {10.2172/1894867},
      url = {https://www.osti.gov/biblio/1894867},
      place = {United States},
      year = {2022},
      month = {10},
      institution = {National Renewable Energy Lab. (NREL)},
   }
```

## WOMBAT in Action

There a few Jupyter notebooks to get users up and running with WOMBAT in the `examples/`
folder, but here are a few highlights:

> **Note**
> In v0.6 the results will diverge significantly under certain modeling conditions from
> past versions due to substantial model upgrades on the backend and new/updated
> features to better specify how repairs are managed.

* Dinwoodie, et al. replication for `wombat` can be found in the
  `examples folder <https://github.com/WISDEM/WOMBAT/blob/main/examples/dinwoodie_validation.ipynb>`_.
* IEA Task 26
  `validation exercise  <https://github.com/WISDEM/WOMBAT/blob/main/examples/iea_26_validation.ipynb>`_.
* Presentations: `slides  <https://github.com/WISDEM/WOMBAT/blob/main/presentation_material/>`_.


## Setup

### Requirements

* Python 3.9 through 3.12

### Environment Setup

Download the latest version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
for the appropriate OS. Follow the remaining
[steps](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
for the appropriate OS version.

Using conda, create a new virtual environment:

```console
conda create -n <environment_name> python=3.11
conda activate <environment_name>
conda install -c anaconda pip

# activate the environment
conda activate <environment_name>

# to deactivate
conda deactivate
```

### Installation

Once in your desired environment, WOMBAT can be installed from PyPI via `pip install`
or from source.

#### Pip

This option is best for those working with the latest release, or including WOMBAT as
a tool in a workflow without the desire to modify the source code.

```console
pip install wombat
```

#### From Source

This option is ideal for users that wish to work with the examples, modify the source
code, and/or contribute back to the project.

Install it directly into an activated virtual environment:

```console
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat
python setup.py install

# Alternatively:
pip install .
```

#### Usage

After installation, the package can imported:

```console
python
import wombat
wombat.__version__
```

For further usage, please see the documentation site at https://wisdem.github.io/WOMBAT.

### Requirements for Contributing to WOMBAT

#### Code Contributions

Code contributors should note that there is both an additional dependency suite for
running the tests and enabling the pre-commit workflow to automatically standardize the
core code formatting principles. In short, the following steps should be taken, but be
sure to read the
[contributor's guide](https://wisdem.github.io/WOMBAT/contributing.html)

```console
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat

# Install the additional dependencies for running the tests and automatic code formatting
pip install -e '.[dev]'

# Enable the pre-commit workflow for automatic code formatting
pre-commit install

# ... contributions and commits ...

# Run the tests and ensure they all pass
pytest tests
```

Basic pre-commit issues that users might encounter and their remedies:

* For any failed run, changes may have been either automatically applied or require
  further edits from the contributor. In either case, after changes have been made,
  contributors will have to rerun `git add <the changed files>` and
  `git commit -m <the commit message>` to restart the pre-commit workflow with the
  applied changes. Once all checks pass, the commit is safe to be pushed.
* `isort`, `black`, or simple file checks failed, but made changes
  * rerun the `add` and `commit` processes as needed until the changes satisfy the checks
* `ruff` failed:
  * Address the errors and rerun the `add` and `commit` processes
* `mypy` has type errors that seem incorrect
  * Double check the typing is in fact as correct as it seems it should be and rerun the
  `add` and `commit` processes
  * If `mypy` simply seems confused with seemingly correct types, the following statement
  can be added above the `mypy` error:
  `assert isinstance(<variable of concern>, <the type you think mypy should be registering>)`
  * If that's still not working, but you are definitely sure the types are correct,
  simply add a `# type ignore` comment at the end of the line. Sometimes `mypy` struggles
  with complex scenarios, or especially with certain `attrs` conventions.

#### Documentation Contributions

```console
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat
pip install -e '.[docs]'
```

Build the site

> **Note**
> You may want to change the "execute_notebooks" parameter in the `docs/_config.yaml`
> file to "off" unless you're updating the coded examples, or they will be run every
> time you build the site.

```console
jupyter-book build docs
```

View the results: `docs/_build/html/index.html`

#### Code and Documentation Contributions

```console
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat
pip install -e '.[all]'
```

### Dependencies

Standard dependencies:

* attrs>=21
* numpy>=1.21
* scipy>=1.8
* pandas>=2
* polars>=0.17
* pyarrow>=10
* jupyterlab>=3
* simpy>=4.0.1
* pyyaml>=6
* geopy>=2.3
* networkx>=2.7
* matplotlib>=3.3
* types-attrs>=19
* types-typed-ast>=1.5
* types-PyYAML>=6
* types-python-dateutil>=2.8

Optional "dev" dependencies:

* pre-commit>=2.20
* isort>=5.10
* pytest>=7
* pytest-cov>=4
* mypy==0.991
* ruff>=0.2
* pyupgrade

Optional "docs" dependencies:

* jupyter-book>=0.15
* myst-nb>=0.16
* myst-parser>=0.17
* linkify-it-py>=2
* sphinx-autodoc-typehints
* sphinxcontrib-autoyaml
* sphinxcontrib-bibtex>=2.4
* sphinxcontrib-spelling>=7
