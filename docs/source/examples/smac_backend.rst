.. _examples_smac_backend:

============
SMAC Backend
============

Hyperactive provides wrappers for SMAC3's (Sequential Model-based Algorithm
Configuration) optimization algorithms, enabling state-of-the-art Bayesian
optimization with Random Forest and Gaussian Process surrogate models.

.. note::

   SMAC must be installed separately:

   .. code-block:: bash

       pip install hyperactive[smac]
       # or
       pip install hyperactive[all_extras]


Available Optimizers
--------------------

SMAC provides three optimization strategies with different surrogate models:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Optimizer
     - Surrogate Model
     - Best For
   * - ``SmacRandomForest``
     - Random Forest
     - Mixed parameter spaces (continuous, categorical, integer)
   * - ``SmacGaussianProcess``
     - Gaussian Process
     - Continuous parameter spaces, small to moderate budgets
   * - ``SmacRandomSearch``
     - None (random sampling)
     - Baseline comparison, high-dimensional spaces


SmacRandomForest
----------------

The flagship SMAC optimizer. Uses a Random Forest surrogate model with
Expected Improvement acquisition function. Handles mixed parameter types natively.

.. code-block:: python

    from hyperactive.opt.smac import SmacRandomForest

    param_space = {
        "C": (0.01, 100.0),           # Float range
        "gamma": (0.0001, 1.0),       # Float range
        "kernel": ["rbf", "linear"],  # Categorical
    }

    optimizer = SmacRandomForest(
        param_space=param_space,
        n_iter=100,
        n_initial_points=10,  # Random points before model-based search
        random_state=42,
        experiment=objective,
    )
    best_params = optimizer.solve()


SmacGaussianProcess
-------------------

Uses a Gaussian Process surrogate model (Matern 5/2 kernel) for sample-efficient
optimization. Best suited for continuous parameter spaces.

.. warning::

   Gaussian Processes scale O(n^3) with observations. Not recommended for
   budgets exceeding 100 evaluations. For mixed or categorical spaces,
   use ``SmacRandomForest`` instead.

.. code-block:: python

    from hyperactive.opt.smac import SmacGaussianProcess

    # Continuous parameters work best with GP
    param_space = {
        "learning_rate": (0.0001, 0.1),
        "weight_decay": (0.0, 0.1),
    }

    optimizer = SmacGaussianProcess(
        param_space=param_space,
        n_iter=50,  # GP is sample-efficient
        random_state=42,
        experiment=objective,
    )
    best_params = optimizer.solve()


SmacRandomSearch
----------------

Pure random search without surrogate modeling. Useful as a baseline or for
high-dimensional spaces where model-based methods struggle.

.. code-block:: python

    from hyperactive.opt.smac import SmacRandomSearch

    optimizer = SmacRandomSearch(
        param_space=param_space,
        n_iter=100,
        random_state=42,
        experiment=objective,
    )
    best_params = optimizer.solve()


Common Parameters
-----------------

All SMAC optimizers share these parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``param_space``
     - Required
     - Search space dictionary with parameter ranges
   * - ``n_iter``
     - 100
     - Number of optimization iterations
   * - ``max_time``
     - None
     - Optional time limit in seconds
   * - ``random_state``
     - None
     - Random seed for reproducibility
   * - ``deterministic``
     - True
     - Whether objective function is deterministic
   * - ``initialize``
     - None
     - Warm start configuration (see below)


Parameter Space Definition
--------------------------

SMAC optimizers support three parameter types:

.. code-block:: python

    param_space = {
        # Float range: both bounds must be float
        "learning_rate": (0.001, 0.1),

        # Integer range: both bounds must be int
        "n_estimators": (10, 500),

        # Categorical: list of choices
        "kernel": ["rbf", "linear", "poly"],
    }

.. note::

   For ambiguous tuples like ``(1, 10)``, Python type determines the parameter
   type. Use ``(1, 10)`` for integer range and ``(1.0, 10.0)`` for float range.


Warm Starting
-------------

Use warm starting to seed optimization with known good configurations:

.. code-block:: python

    optimizer = SmacRandomForest(
        param_space=param_space,
        n_iter=100,
        initialize={
            "warm_start": [
                {"C": 1.0, "gamma": 0.1, "kernel": "rbf"},
                {"C": 10.0, "gamma": 0.01, "kernel": "linear"},
            ]
        },
        experiment=objective,
    )


When to Use SMAC Backend
------------------------

The SMAC backend is useful when you need:

- **State-of-the-art Bayesian optimization** with proven surrogate models
- **Native handling of mixed parameter spaces** (Random Forest handles categorical parameters well)
- **Sample-efficient optimization** for expensive function evaluations
- **Hyperparameter optimization** following AutoML best practices
- **Reproducible results** in scientific experiments

Choose ``SmacRandomForest`` when:

- Your search space has mixed parameter types
- You have 50+ evaluations budget
- Parameters interact in complex ways

Choose ``SmacGaussianProcess`` when:

- All parameters are continuous
- Budget is small (10-50 evaluations)
- You need uncertainty estimates

Choose ``SmacRandomSearch`` when:

- You need a baseline for comparison
- Search space is high-dimensional (>20 parameters)
- Evaluations are cheap and parallelizable


References
----------

- `SMAC3 Documentation <https://automl.github.io/SMAC3/main/>`_
- Lindauer, M., et al. (2022). SMAC3: A Versatile Bayesian Optimization
  Package for Hyperparameter Optimization. JMLR.
