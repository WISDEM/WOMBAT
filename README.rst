WOMBAT: Windfarm Operations & Maintenance cost-Benefit Analysis Tool
====================================================================

.. image:: https://img.shields.io/badge/DOI-10.2172%2F1894867-brightgreen?link=https://doi.org/10.2172/1894867
   :target: https://www.osti.gov/biblio/1894867

.. image:: https://badge.fury.io/py/wombat.svg
   :target: https://badge.fury.io/py/wombat

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/WISDEM/WOMBAT/main?filepath=examples

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0


.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/



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

If you use this library please cite our NREL Technical Report:

.. code-block:: bibtex

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


WOMBAT in Action
----------------

There a few Jupyter notebooks to get users up and running with WOMBAT in the `examples/`
folder, but here are a few highlights:

* Dinwoodie, et al. replication for `wombat` can be found in the
  `examples folder <https://github.com/WISDEM/WOMBAT/blob/main/examples/dinwoodie_validation.ipynb>`_.
* IEA Task 26
  `validation exercise  <https://github.com/WISDEM/WOMBAT/blob/main/examples/iea_26_validation.ipynb>`_.
* Presentations: `slides  <https://github.com/WISDEM/WOMBAT/blob/main/presentation_material/>`_.

Setup
-----

Requirements
~~~~~~~~~~~~

* Python 3.8+, see the next section for more.

Environment Setup
~~~~~~~~~~~~~~~~~

Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
for the appropriate OS version.

Using conda, create a new virtual environment:

.. code-block:: console

   conda create -n <environment_name> python=3.8 --no-default-packages
   conda activate <environment_name>
   conda install -c anaconda pip

   # to deactivate
   conda deactivate



Installation
~~~~~~~~~~~~


Pip
~~~

.. code-block:: console

   pip install wombat


From Source
~~~~~~~~~~~

Install it directly into an activated virtual environment:

.. code-block:: console

   git clone https://github.com/WISDEM/WOMBAT.git
   cd wombat
   python setup.py install


or if you will be contributing:

.. code-block:: console

   git clone https://github.com/WISDEM/WOMBAT.git
   cd wombat
   pip install -e '.[dev]'


Required for automatic code formatting!

.. code-block:: console

   pre-commit install


or for documentation:

.. code-block:: console

   git clone https://github.com/WISDEM/WOMBAT.git
   cd wombat
   pip install -e '.[docs]'


Build the site

NOTE: You may want to change the "execute_notebook" parameter in the `conf.py` file to
"off" unless you're updating the coded examples or they will be run every time you
build the site.

.. code-block:: console

   cd docs/
   make html


View the results: `docs/_build/html/index.html`

or both at once:

.. code-block:: console

   git clone https://github.com/WISDEM/WOMBAT.git
   cd wombat
   pip install -e '.[all]'


Usage
-----

After installation, the package can imported:

.. code-block:: console

   python
   import wombat
   wombat.__version__
