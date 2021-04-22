# WOMBAT: Winfarm Operations and Maintenance cost-Benefit Analysis Tool

This library provides a tool to simulate the operation and maintenance phase (O&M) of
distributed, land-based, and offshore windfarms using a discrete event simultaion
framework.

`WOMBAT` is written around the [`Simpy`](https://gitlab.com/team-simpy/simpy) framework
for discrete event simulation framework. Additionally, this is supported using a
flexible and modular object-oriented code base, which enables the modeling of
arbitrarily large (or small) windfarms with as many or as few failure and maintenance
tasks that can be encoded.

# Setup

## Requirements

* Python 3.7+

## Installation

Download the latest version of [Miniconda](<https://docs.conda.io/en/latest/miniconda.html>)
for the appropriate OS. Follow the remaining [steps](<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>)
for the appropriate OS version.

Using conda, create a new virtual environment:
```text
$ conda create -n <environment_name> python=3.7 --no-default-packages
$ conda activate <environment_name>
$ conda install -c anaconda pip
# to deactivate: conda deactivate
```

Install it directly into an activated virtual environment:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ python setup.py install
# $ pip install wombat - coming soon!
```

or if you will be contributing:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ pip install -e '.[dev]'
# Required for automatic code formatting!
$ pre-commit install
```

or for documentation:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ pip install -e '.[docs]'
```
# Usage

After installation, the package can imported:

```text
$ python
>>> import wombat
>>> wombat.__version__
```
