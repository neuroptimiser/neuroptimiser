Architecture
============

The NeurOptimiser architecture defines a fully asynchronous, spike-driven metaheuristic framework where distributed spiking neural activity directly governs the optimisation process. This design integrates multiple neuromorphic principles into a coherent and modular architecture composed of hierarchical components.

NeurOptimiser Global Architecture
---------------------------------

The global architecture consists of a set of distributed *Neuromorphic Heuristic Units* (NHUs), together with auxiliary modules that coordinate information exchange at different levels. Each NHU operates independently, processing candidate solutions via internal neural dynamics and local heuristic rules triggered upon spiking.

The following diagram presents the complete NeurOptimiser architecture:

.. image:: _static/images/nheuristic_algorithm_h.pdf
    :align: center
    :width: 75%
    :alt: NeurOptimiser Architecture

This system is composed of:

- **Neuromorphic Heuristic Units (NHUs)** independent neuromorphic agents that encode, evaluate, and update candidate solutions.
- **Tensor Contraction Layer** regulates distributed spike-based activation by contracting the spike matrix across the connectivity structure defined by the neuron topology.
- **Neighbour Manager** extracts and filters local neighbourhood information from the population-wide solution pool, providing decentralised information exchange for neighbourhood-based operators.
- **High-Level Selector** identifies the global best solution across all NHUs based on the currently shared broadcasted information.

Each NHU encapsulates its own spiking dynamics, perturbation mechanisms, and local evaluation logic, fully embedded within its spiking behaviour. The architecture permits the framework to scale efficiently across problem dimensionality, population size, and hardware configurations, supporting multiple neural models (e.g., linear, Izhikevich) and topologies (e.g., ring, full, random).

Neuromorphic Heuristic Unit (NHU)
----------------------------------

At the core of the optimisation process lies the NHU, which integrates the essential spiking computation and spike-triggered heuristic variation mechanisms. Each NHU operates autonomously, embedding both the state evolution and heuristic adaptation within spike-triggered transitions.

The internal structure of the NHU is depicted below:

.. image:: _static/images/nheuristic_unit_h.pdf
    :align: center
    :width: 75%
    :alt: Neuromorphic Heuristic Unit

The NHU is composed of the following processes:

- **Spiking Core** is a dynamic neural subsystem implementing continuous-time evolution models (e.g., linear attractors, Izhikevich neurons) that encode the candidate solution state.
- **Selector** is a local objective evaluation component performing fitness computation and greedy selection of best-so-far solutions.
- **Spiking Handler** manages conversion of internal spike signals into shared activation matrices used for decentralised inter-unit excitation.
- **Receiver** retrieves neighbouring solutions and fitness values based on the pre-defined neighbourhood structure.
- **Sender** broadcasts locally updated solutions and fitness values to the global matrices used for coordination.

Each NHU operates asynchronously, continuously processing available inputs, updating its internal state, applying heuristic perturbations upon spiking, and contributing to the emergent global optimisation process. This fully distributed execution allows the framework to scale without requiring any centralised control or synchronisation.

----

For further details on the NeurOptimiser model dynamics, spiking core equations, and heuristic operators, please refer to the corresponding implementation modules at :doc:`api` and experimental validation presented in the paper in [1]_.

.. rubric:: References

.. [1] Jorge M. Cruz-Duarte and El-Ghazali Talbi. **NeurOptimiser: A Neuromorphic Metaheuristic Framework for Asynchronous Spike-Driven Optimisation**, in *WBO 2025, Springer Proceedings*, 2025. [Submitted]
