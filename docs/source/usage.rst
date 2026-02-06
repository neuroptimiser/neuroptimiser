Usage
=====

Quick Start
-----------

To begin using `neuroptimiser`, follow these steps:

1. **Clone the Repository**:

    .. code-block:: bash

        git clone https://github.com/neuroptimiser/neuroptimiser.git
        cd neuroptimiser

2. **Create and Activate a Virtual Environment** (Python 3.10 recommended):

    .. code-block:: bash

        python3.10 -m venv .venv
        source .venv/bin/activate  # On Unix/Linux/macOS
        # .venv\Scripts\activate  # On Windows

3. **Install Dependencies**:

    Using pip:

    .. code-block:: bash

        pip install -e .
        # or from PyPI
        pip install neuroptimiser

    Using uv (recommended):

    .. code-block:: bash

        uv pip install -e .
        # or from PyPI
        uv pip install neuroptimiser

    You can also use the provided ``Makefile`` for additional installation options and commands.

4. **Prevent "Too many open files" error** (if needed):

    On Unix/Linux/macOS:

    .. code-block:: bash

        ulimit -n 65536

    On Windows, this is typically not needed as Windows has higher default limits.

Example Usage
-------------

Here's a simple example to get started:

.. code-block:: python

    from neuroptimiser import NeurOptimiser
    import numpy as np

    problem_function = lambda x: np.linalg.norm(x)
    problem_bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

    optimiser = NeurOptimiser()

    optimiser.solve(
        obj_func=problem_function,
        search_space=problem_bounds,
        debug_mode=True,
        num_iterations=1000,
    )


Examples
--------

Explore the following examples to understand how to utilise `neuroptimiser`:

.. toctree::
    :maxdepth: 1

    examples/Example0
    examples/Example1
    examples/Example2

Further Reading
---------------

For more detailed information, refer to the following sections:

- :doc:`installation`
- :doc:`api`
- :doc:`contributing`