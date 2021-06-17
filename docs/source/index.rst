WOMBAT - Windfarm Operations and Maintenance cost-Benefit Analysis Tool
=======================================================================

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

.. image:: images/code_hierarchy.png

As for how a windfarm is simulated, the below image represents the flow of events as
they occur within the model.

.. image:: images/simulation_architecture.png


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


License
-------

Apache 2.0; please see the repository for license information:
https://github.com/WISDEM/WOMBAT/blob/master/LICENSE