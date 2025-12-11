.. _troubleshooting_installation:

===================
Installation Issues
===================

ImportError: No module named 'hyperactive'
------------------------------------------

**Cause**: Hyperactive is not installed or installed in a different environment.

**Solution**:

.. code-block:: bash

    pip install hyperactive

    # Or with extras
    pip install hyperactive[all_extras]

Verify installation:

.. code-block:: bash

    python -c "import hyperactive; print(hyperactive.__version__)"


ImportError: cannot import name 'Hyperactive'
---------------------------------------------

**Cause**: You're using v4 code with Hyperactive v5. The ``Hyperactive`` class
was removed in v5.

**Solution**: Update your imports. See :ref:`user_guide_migration` for details.

.. code-block:: python

    # Old (v4)
    from hyperactive import Hyperactive

    # New (v5)
    from hyperactive.opt.gfo import HillClimbing


Missing Optional Dependencies
-----------------------------

**Cause**: Some features require additional packages.

**Solution**: Install the appropriate extras:

.. code-block:: bash

    # For scikit-learn integration
    pip install hyperactive[sklearn-integration]

    # For Optuna backend
    pip install hyperactive[optuna]

    # For all extras
    pip install hyperactive[all_extras]
