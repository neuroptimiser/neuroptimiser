API Reference
=============

The NeurOptimiser framework exposes several modules to build and execute neuromorphic metaheuristics. This API reference provides documentation for all public components, grouped by their functional roles.

For the general architecture and design principles, refer to the :doc:`architecture` section. The main entry points for most users are the solvers and core models. Internal modules provide access to lower-level processes, models, and utilities.

The source code for `NeurOptimiser is on GitHub <https://github.com/neuroptimiser/neuroptimiser>`__.

.. important:: Please, consider that this API is under active development, and some components may change in future releases.


Modules Overview
----------------

- :doc:`api-solvers` – High-level solver classes.
- :doc:`api-core` – Core components, processes, and models.
- :doc:`api-utils` – Utility functions and helpers.

----

.. toctree::
   :maxdepth: 3
   :caption: Modules

   api-solvers
   api-core
   api-utils
