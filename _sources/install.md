# Installation

## Create a conda environment

Download the latest version of [Miniconda](<https://docs.conda.io/en/latest/miniconda.html>)
for the appropriate OS. Follow the remaining [steps](<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>)
for the appropriate OS version.

Using conda, create a new virtual environment, replacing `<environment_name>` with a name
of your choosing (without spaces):
```text
$ conda create -n <environment_name> python=3.x  # use your preferred 3.9-3.12 version
$ conda activate <environment_name>
```
You can now use ``conda activate <environment_name>`` to enter the environment and
``conda deactivate`` to exit the environment.


## Installing

```{note}
The folowing all asssume you are in your `wombat` environment!
```

### Pip

WOMBAT is listed on PyPI under `wombat`, and can be installed for users that don't
intend to modify the code, by running the following:

```bash
pip install wombat
```

### From Source

If you plan to read through the code, in addition to running simulations, it is
recommended to clone repository, and install the local version.

```{note}
It has been noted that some Windows users have had issues with accessing the included
data libraries using the following, so if issues arise, uninstall wombat, and reinstall
using the `pip install -e '.[dev]'` prompt in the next section
```

```bash
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat
pip install wombat/
```

### For contributors

Anyone seeking to work modify the code and contribute to the repository should follow
the below prompt. This repository relies on automatic code formatting and linting
provided through the pre-commit framework, and any contributors should have this
functionality run and pass before submitting pull requests.

```bash
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat
pip install -e '.[dev]'  # some users may need double quotes here, not single quotes
# Required for automatic code formatting!
pre-commit install
```

#### Running the tests

In addition, pytest is used to manage the testing framework, and can be run using the
following to ensure that contributions don't break the existing tests. This will produce
the results of the tests, including code coverage results, if the tests pass.

```bash
# From the top level of the repository
pytest tests/
```

### For documentation

Additionally, for users that wish to modify the documentation or build the documentation
site locally, the following prompt with will install the required documentation building
packages.

```bash
git clone https://github.com/WISDEM/WOMBAT.git
cd wombat
pip install -e '.[docs]'
```

#### Build the documentation site

```bash
jupyter-book build docs
```

Now, the documentation site should be able to be viewed locally at
docs/_build/html/index.html
