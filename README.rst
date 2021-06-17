====================================================================
WOMBAT: Windfarm Operations & Maintenance cost-Benefit Analysis Tool
====================================================================

This library provides a tool to simulate the operation and maintenance phase (O&M) of
distributed, land-based, and offshore windfarms using a discrete event simultaion
framework.

WOMBAT is written around the `SimPy <https://gitlab.com/team-simpy/simpy>`_ framework
for discrete event simulation framework. Additionally, this is supported using a
flexible and modular object-oriented code base, which enables the modeling of
arbitrarily large (or small) windfarms with as many or as few failure and maintenance
tasks that can be encoded.

Please note that this is still heavily under development, so you may find some functionality
to be incomplete at the current moment, but rest assured the functionality is expanding.
With that said, it would be greatly appreciated for issues or PRs to be submitted for
any improvements at all, from fixing typos (guaranteed to be a few) to features to
testing (coming FY22!).

WOMBAT in Action
================

There a few Jupyter notebooks to get users up and running with WOMBAT in the `examples/`
folder, but here are a few highlights:

* Dinwoodie, et al. replication for `wombat` can be found in the
`examples folder <https://github.com/WISDEM/WOMBAT/blob/main/examples/dinwoodie_validation.ipynb>`_.
* IEA Task 26
`validation exercise  <https://github.com/WISDEM/WOMBAT/blob/main/examples/iea_26_validation.ipynb>`_.
* Presentations: `slides  <https://github.com/WISDEM/WOMBAT/blob/main/presentation_material/>`_.

Setup
=====

Requirements
------------

* Python 3.7+, see the next section for more.

Environment Setup
-----------------

Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
   for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
   for the appropriate OS version.

Using conda, create a new virtual environment:
```text
$ conda create -n <environment_name> python=3.8 --no-default-packages
$ conda activate <environment_name>
$ conda install -c anaconda pip
# to deactivate: conda deactivate
```


Installation
------------


Pip
---

```text
$ pip install wombat
```

NOTE: For now, you will have to download the data separetely if you're going to be
using the "dinwoodie" or "iea_26" data libraries. This will amended before the end of
the year.


From Source
-----------

Install it directly into an activated virtual environment:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ python setup.py install
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

# Build the site
# NOTE: You may want to change the "execute_notebook" parameter in the `conf.py` file to
# "off" unless you're updating the coded examples or they will be run every time you
# build the site

$ cd docs/
$ make html

# view the results: `docs/_build/html/index.html`
```

or both at once:

```text
$ git clone https://github.com/WISDEM/WOMBAT.git
$ cd wombat
$ pip install -e '.[all]'
```

Usage
=====

After installation, the package can imported:

```text
$ python
>>> import wombat
>>> wombat.__version__
```
