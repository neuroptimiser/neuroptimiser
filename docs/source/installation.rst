Installation
============

Neuroptimiser is available on PyPI and can be installed via pip or uv.

----

System Requirements
-------------------

- **Python 3.10** (tested and recommended version)
- Lava-NC environment configured
- Operating System: Linux, macOS, or Windows
- A virtual environment is strongly recommended to isolate dependencies.
- `uv from astral <https://docs.astral.sh/uv/>`__ is strongly recommended for managing the project environment.

----

Installation Methods
---------------------

**1. Installing from PyPI (Recommended)**

You can install Neuroptimiser directly from PyPI:

.. code-block:: bash

    # Using pip
    pip install neuroptimiser

    # Using uv (recommended)
    uv pip install neuroptimiser

**2. Installing from GitHub (development version)**

If you prefer to install the latest development version directly from the GitHub repository, you can use:

.. code-block:: bash

    pip install git+https://github.com/neuroptimiser/neuroptimiser.git

Alternatively, you can clone the repository and install in *editable mode* for local development:

.. code-block:: bash

    git clone https://github.com/neuroptimiser/neuroptimiser.git
    cd neuroptimiser

    # Using pip
    pip install -e .

    # Using uv (recommended)
    uv pip install -e .

You can also use the provided ``Makefile`` for additional installation options and commands.


----

Known Issues
------------

**"Too many open files" error**

On some systems, you may encounter this error during execution. To fix it:

**Unix/Linux/macOS:**

.. code-block:: bash

    ulimit -n 65536

**Windows (PowerShell):**

.. code-block:: powershell

    # No direct equivalent - typically not needed on Windows
    # If issues persist, check system file handle limits via registry

**Windows (Command Prompt):**

.. code-block:: bat

    REM Windows typically has higher default limits
    REM If needed, adjust via Registry Editor or contact system administrator

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