.. _optimizers_ref:

Optimization backends
=====================

The :mod:`hyperactive.opt` module contains optimization algorithms for hyperparameter tuning.

All optimizers inherit from :class:`~hyperactive.base.BaseOptimizer` and share the same interface:
the ``solve()`` method to run optimization, and configuration via the ``experiment`` and ``search_space`` parameters.

Hyperactive provides optimizers from three backends:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Backend
     - Description
   * - :doc:`gfo`
     - Native gradient-free optimization algorithms (21 optimizers)
   * - :doc:`optuna`
     - Interface to Optuna's samplers (8 optimizers)
   * - :doc:`sklearn`
     - sklearn-compatible search interfaces (2 optimizers)

.. toctree::
    :maxdepth: 2

    gfo
    optuna
    sklearn
