Installation
============

.. caution::
    This instructions are generic and will be revised soon.

To install neurOptimiser, you can use pip, which is the recommended way to install Python packages. Open your terminal or command prompt and run the following command:

.. code-block:: bash

    pip install neuroptimiser


If you prefer to install the latest development version directly from the GitHub repository, you can use:

.. code-block:: bash

    pip install git+https://github.com/jcrvz/neuroptimiser.git

If you want to install neurOptimiser in a virtual environment, you can follow these steps:

1. Create a new virtual environment:

    .. code-block:: bash

        python -m venv neuroptimiser-env

2. Activate the virtual environment:

    - On Windows:

      .. code-block:: bash

         neuroptimiser-env\Scripts\activate

    - On macOS and Linux:

      .. code-block:: bash

         source neuroptimiser-env/bin/activate

3. Install neurOptimiser within the activated virtual environment:

   .. code-block:: bash

        pip install neuroptimiser

4. To verify the installation, you can run a simple Python script:

   .. code-block:: python

        >>> import neuroptimiser
        >>> print("Neuroptimiser installed successfully!")

5. If you need to uninstall neurOptimiser, you can do so with:

.. code-block:: bash

        pip uninstall neuroptimiser

6. For development purposes, if you want to clone the repository and install it in editable mode, you can do:

.. code-block:: bash

    git clone https://github.com/jcrvz/neuroptimiser
    cd neuroptimiser
    pip install -e .

This will allow you to make changes to the code and have them reflected immediately without needing to reinstall the package.