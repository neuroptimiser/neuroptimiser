.. neuroptimiser documentation master file, created by
   sphinx-quickstart on Wed May 28 14:16:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NeurOptimiser's Documentation
=====================================

**NeurOptimiser** is a neuromorphic optimisation framework built on top of the Lava-NC framework [1]_. It combines spiking neural dynamics with heuristic search principles, enabling the design, simulation, and evaluation of decentralised, asynchronous metaheuristic processes inspired by biological computation. This documentation provides detailed information for users, developers, and researchers aiming to utilise, extend, or contribute to the NeurOptimiser framework.

For more information, visit NeurOptimiser on `arXiv <https://doi.org/10.48550/arXiv.2507.08320>`__ [2]_ and `GitHub <https://github.com/neuroptimiser/neuroptimiser>`__.

.. note::
   This project is under active development.

Documentation Overview
----------------------
The documentation is organised into the following sections:

- :doc:`Installation Guide <installation>`: Instructions for installing NeurOptimiser and setting up the development environment.
- :doc:`Usage Guide <usage>`: Examples and instructions on configuring and running optimisation tasks.
- :doc:`Architecture <architecture>`: Detailed description of the internal modules, components, and computational flow.
- :doc:`API Reference <api>`: Automatically generated documentation for public classes, functions, and modules.
- :doc:`Contributing Guide <contributing>`: Information for contributors, including coding standards and contribution workflow.

Getting Started
----------------
To start using NeurOptimiser:

1. Install the framework locally or via package manager (see the Installation Guide).
2. Import and instantiate `NeurOptimiser` or its modular components.
3. Configure the optimisation problem, spiking model, and metaheuristic parameters.
4. Execute and analyse the simulation results.

If you are new to neuromorphic computing or metaheuristic optimisation, we recommend reviewing the Usage Guide for step-by-step examples.

----

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Architecture
   :hidden:

   architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
    :maxdepth: 2
    :caption: Events
    :hidden:

    events/index

.. toctree::
   :maxdepth: 2
   :caption: Contribution Guide
   :hidden:

   contributing

References
----------
.. [1] Intel Neuromorphic Computing Lab. **Lava-NC: A software framework for neuromorphic computing**, 2023. Available at: https://lava-nc.org
.. [2] Cruz-Duarte JM, Talbi EG. **NeurOptimisation: The Spiking Way to Evolve**. arXiv preprint arXiv:2507.08320, 2025 Jul 11. Available at: https://doi.org/10.48550/arXiv.2507.08320