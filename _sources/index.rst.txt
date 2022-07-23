WOMBAT - Windfarm Operations and Maintenance cost-Benefit Analysis Tool
=======================================================================

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


Overview
--------

The WOMBAT framework is designed to provide an open source tool adhering to FLOSS principles
for the windfarm lifecycle research community. Specifically, WOMBAT is meant to serve as
a what-if, or scenario-based, simulation tool, so that you can model the trade-offs in
decision making for the operations and maintenance phase of a windfarm.

Please note that for the current version, the documentation is fairly sparse outside of
the core API documentation. This will be updated, along with the model by the end of the
summer when more development is completed. By that point the model should include more
functionality surrounding unscheduled maintenance in addition to added documentation.

For any questions, feel free to open up an issue in the repository or email:
rob.hammond@nrel.gov.

What's New in v0.5?
-------------------
- Tow-to-port for offshore wind farms!
- More metrics!
- Bug fixes!
- See the changelog for full details


The Model in 30 Seconds Or Less
-------------------------------

In general, the model has 2 overarching branches: the windfarm itself, and the
simulation environment. For the wind farm model we can control the varying assets
(or system in the code) as well as the components (or subassemblies in the code). This
separation allows for each turbine, cable, or substation component to have its own unique
failure and maintenance models.

As for the environment, this is where the discrete event simulation itself happens in
addition to logging, repair logic, and other necessary modeling pieces. The image
below provides a more visual representation of this description.


High Level Architecture
^^^^^^^^^^^^^^^^^^^^^^^

The code is largely broken up into two categories: the windfarm and objects contained
within it, and the simulation and simulation evironment components. The windfarm is
composed of systems: substation(s), cables, and turbines, and each of those systems is
composed of subassemblies (a conglomerate of components). For the simulation environment,
we consider all of the pieces that allow the simulation to happen such as the API,
servicing equipment, repair manager to hold and pass on tasks, and results post-processing.

.. image:: images/high_level_diagram.svg

The following section describes how a windfarm is simulated and the flow of events as
they occur within the model.

Simulation Architecture
^^^^^^^^^^^^^^^^^^^^^^^

In the diagram below, we demonstrate the lifecycle of the simulation through the
lifecycle of a single failure.

1) The maximum length of the simulation is defined by the amount of wind and wave
   timeseries data is provided for the simulation.
2) Each subassemly failure model is a random sampling from a Weibull distribution, so
   for the sake of clarity we'll consider this to be a catastrophic drivetrain failure.
   When the timeout (time to failure) is reached in the simulation, the subassembly's operating level is
   reduced to 0%, and a message is passed to the turbine level (the overarching system
   model).
3) From there, the turbine will shut off, and signal to all other subassembly models to
   not accrue time for their respective maintenance and failure timeouts. Then, the
   turbine creates a repair request and passes that to the repair manager.
4) The repair manager will store the submitted task and depending on the servicing
   equipment's maintenance strategy, a crane or vessel will be called to site.
5) The vessel (or crane) will mobilize and accrue time there, all the while, the turbine
   and wind power plant will accrue downtime. When the servicing equipment arrives at
   the site it will operate according to its operation limits, current weather
   conditions, and site/equipment-specific working hours and continue to log costs and
   downtime.
6) When the servicing is complete, the subassembly will be placed back to good-as-new
   condition and the turbine will be reset to operating. From there all the turbine's
   and drivetrain's failure and maintenance models be turned back on, and the simulation
   will continue on in the same manner until it reaches it's user- or weather-defined
   ending point.

.. image:: images/simulation_diagram.png


Welcome
-------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   presentations
   team



Using WOMBAT
------------

.. toctree::
   :maxdepth: 2
   :caption: Working with the code base

   examples/index
   API/index
   changelog


License
-------

Apache 2.0; please see the repository for license information:
https://github.com/WISDEM/WOMBAT/blob/master/LICENSE
