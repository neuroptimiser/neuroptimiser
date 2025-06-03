Solvers
=======

The solvers module defines the abstract interfaces and concrete implementations used to perform the optimisation process within the Neuroptimiser framework.

The solver architecture is designed following the modular structure described in the :doc:`architecture` section. It integrates multiple neuromorphic heuristic units to drive the search process asynchronously.


.. currentmodule:: neuroptimiser.solvers

Abstract Solver
---------------

The base class providing the general interface for solver implementations.

.. autoclass:: AbstractSolver
   :members:
   :undoc-members:
   :show-inheritance:

NeurOptimiser
-------------

The concrete solver implementation integrating neuromorphic heuristic units, local dynamics, and spike-driven transitions.

.. autoclass:: NeurOptimiser
   :members:
   :undoc-members:
   :show-inheritance:

