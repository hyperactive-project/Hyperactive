.. _installation:

============
Installation
============

Hyperactive can be installed via pip and supports Python |python_version_range|.

Installing Hyperactive
----------------------

Basic Installation
^^^^^^^^^^^^^^^^^^

Install Hyperactive from PyPI using pip:

.. code-block:: bash

    pip install hyperactive

This installs Hyperactive with its core dependencies, which is sufficient for most use cases
including scikit-learn integration.


Installation with Extras
^^^^^^^^^^^^^^^^^^^^^^^^

For additional functionality, you can install optional extras:

.. code-block:: bash

    # Full installation with all extras (Optuna, PyTorch Lightning, etc.)
    pip install hyperactive[all_extras]

    # Scikit-learn integration (included by default)
    pip install hyperactive[sklearn-integration]

    # Sktime/skpro integration for time series
    pip install hyperactive[sktime-integration]


Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

To install Hyperactive for development (from source):

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/SimonBlanke/Hyperactive.git
    cd Hyperactive

    # Install in development mode with test dependencies
    pip install -e ".[test]"

    # Or install with all development dependencies
    pip install -e ".[test,docs]"


Dependencies
------------

Core Dependencies
^^^^^^^^^^^^^^^^^

Hyperactive requires the following packages (automatically installed):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``numpy >= 1.18.1``
     - Numerical operations and array handling
   * - ``pandas < 3.0.0``
     - Data manipulation and results handling
   * - ``tqdm >= 4.48.0``
     - Progress bars during optimization
   * - ``gradient-free-optimizers >= 1.2.4``
     - Core optimization algorithms
   * - ``scikit-base < 1.0.0``
     - Base classes for sklearn-like interfaces
   * - ``scikit-learn < 1.8.0``
     - Machine learning integration


Optional Dependencies
---------------------

Depending on your use case, you may want to install additional packages:

Optuna Backend
^^^^^^^^^^^^^^

For Optuna-based optimizers (TPE, CMA-ES, NSGA-II, etc.):

.. code-block:: bash

    pip install optuna

Or include it via the ``all_extras`` option:

.. code-block:: bash

    pip install hyperactive[all_extras]


Time Series (sktime/skpro)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For time series forecasting and probabilistic prediction:

.. code-block:: bash

    pip install hyperactive[sktime-integration]

This installs ``sktime`` and ``skpro`` for time series optimization experiments.

.. note::

   Sktime integration requires Python < 3.14 due to sktime's current compatibility.


PyTorch Lightning
^^^^^^^^^^^^^^^^^

For deep learning hyperparameter optimization:

.. code-block:: bash

    pip install hyperactive[all_extras]
    # or
    pip install lightning

.. note::

   PyTorch/Lightning requires Python < 3.14 for full compatibility.


Verifying Installation
----------------------

After installation, verify that Hyperactive is working correctly:

.. literalinclude:: _snippets/installation/verify_installation.py
   :language: python
   :start-after: # [start:verify_installation]
   :end-before: # [end:verify_installation]


Python Version Support
----------------------

Hyperactive officially supports Python |python_versions_list|.

.. note::

   The supported Python versions are automatically extracted from the project's
   ``pyproject.toml`` classifiers.

   Some optional integrations (sktime, PyTorch) may have more restrictive
   Python version requirements. Check the specific package documentation
   for details.


Troubleshooting
---------------

Common Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**ImportError: No module named 'gradient_free_optimizers'**

This usually means the installation was incomplete. Try:

.. code-block:: bash

    pip install --upgrade hyperactive

**MemoryError during optimization**

Sequential model-based optimizers (Bayesian, TPE) can use significant memory
for large search spaces. Reduce your search space size or use a simpler optimizer
like ``RandomSearch`` or ``HillClimbing``.

**Pickle errors with multiprocessing**

Ensure all objects in your search space are serializable (no lambdas, closures,
or bound methods). Use top-level functions and basic Python types.

For more help, see the `GitHub Issues <https://github.com/SimonBlanke/Hyperactive/issues>`_.


Using Older Versions
--------------------

If you need to use Hyperactive v4, you can install a specific version:

.. code-block:: bash

    pip install hyperactive==4.8.0

Documentation for v4 is available at:
`Legacy Documentation (v4) <https://simonblanke.github.io/hyperactive-documentation/5.0/>`_
