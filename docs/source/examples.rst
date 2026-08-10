.. _examples:

========
Examples
========

This section provides a collection of examples demonstrating Hyperactive's capabilities.
All examples are available in the
`examples directory <https://github.com/SimonBlanke/Hyperactive/tree/master/examples>`_
on GitHub.

.. toctree::
   :maxdepth: 1

   examples/general
   examples/local_search
   examples/global_search
   examples/population_based
   examples/sequential_model_based
   examples/optuna_backend
   examples/scipy_backend
   examples/sklearn_backend
   examples/integrations
   examples/other
   examples/interactive_tutorial


Overview
--------

Hyperactive provides examples for all optimization algorithms and integration
patterns. The examples are organized by algorithm category:


Gradient-Free Optimizers (GFO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`examples_general`
    Basic examples showing custom function optimization and sklearn model tuning.

:ref:`examples_local_search`
    Hill Climbing, Simulated Annealing, Downhill Simplex, and other local
    search methods that explore by making incremental moves.

:ref:`examples_global_search`
    Random Search, Grid Search, Powell's Method, and other algorithms that
    explore the search space more broadly.

:ref:`examples_population_based`
    Particle Swarm, Genetic Algorithm, Evolution Strategy, and other
    nature-inspired population methods.

:ref:`examples_sequential_model_based`
    Bayesian Optimization, Tree-Parzen Estimators, and other model-based
    methods that learn from previous evaluations.


Backend Examples
^^^^^^^^^^^^^^^^

:ref:`examples_optuna_backend`
    Examples using Optuna's samplers including TPE, CMA-ES, NSGA-II/III,
    and Gaussian Process optimization.

:ref:`examples_scipy_backend`
    Examples using scipy.optimize algorithms including Differential Evolution,
    Dual Annealing, Basin-hopping, SHGO, DIRECT, Nelder-Mead, and Powell.

:ref:`examples_sklearn_backend`
    Scikit-learn compatible interfaces as drop-in replacements for
    GridSearchCV and RandomizedSearchCV.


Integration Examples
^^^^^^^^^^^^^^^^^^^^

:ref:`examples_integrations`
    Time series optimization with sktime and other framework integrations.


Advanced Topics
^^^^^^^^^^^^^^^

:ref:`examples_other`
    Advanced usage patterns including warm starting and optimizer comparison.

:ref:`examples_interactive_tutorial`
    Comprehensive Jupyter notebook tutorial covering all Hyperactive features.
