Installation
============

Neuroptimiser is under active development and will soon be available via PyPI. Until the first official release is published, installation is supported through the GitHub repository.

.. attention::
   These installation instructions apply to the current development version. **PyPI installation will be added once the first stable release is published.**

----

System Requirements
-------------------

- Python >= 3.10
- Operating System: Linux, macOS, or Windows (not tested yet)
- A virtual environment is strongly recommended to isolate dependencies.
- `uv from astral <https://docs.astral.sh/uv/>`__ is strongly recommended for managing the project environment.

----

Installation Methods
---------------------

**1. Installing from GitHub (development version)**

If you prefer to install the latest development version directly from the GitHub repository, you can use:

.. code-block:: bash

    pip install git+https://github.com/neuroptimiser/neuroptimiser.git

Alternatively, you can clone the repository and install in *editable mode* for local development:

.. code-block:: bash

    git clone https://github.com/neuroptimiser/neuroptimiser.git
    cd neuroptimiser
    pip install -e .

**2. Installing from PyPI (available soon)**

The package can be installed via:

.. code-block:: bash

    pip install neuroptimiser

----

Optional: Virtual Environment Setup
-----------------------------------

Although not strictly required, it is highly recommended to create a virtual environment to isolate the installation.

For example:


If you want to install neurOptimiser in a virtual environment, you can follow these steps:

.. code-block:: bash

    python -m venv .venv

    # Linux or macOS
    source .venv/bin/activate

    # On Windows
    .venv\Scripts\activate

Then proceed with one of the installation methods above.

Verifying Installation
-----------------------

To verify the installation, you can run the following Python commands:

.. code-block:: python

    >>> import neuroptimiser
    >>> print("Neuroptimiser installed successfully")

----


Uninstalling
------------

To uninstall Neuroptimiser:

.. code-block:: bash

    pip uninstall neuroptimiser