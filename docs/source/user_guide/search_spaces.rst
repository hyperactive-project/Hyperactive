.. _user_guide_search_spaces:

=========================
Search Space Best Practices
=========================

This guide covers how to design effective search spaces for hyperparameter optimization.
A well-designed search space can significantly improve optimization results and efficiency.


Understanding Search Spaces
---------------------------

A search space defines the possible values for each parameter. Hyperactive samples
from these values during optimization. The quality of your search space directly
affects:

- **Optimization speed**: Smaller, targeted spaces converge faster
- **Solution quality**: Including good values is essential for finding them
- **Memory usage**: Very large spaces can cause memory issues with some optimizers


Defining Search Spaces
----------------------

Basic Structure
^^^^^^^^^^^^^^^

Search spaces are Python dictionaries mapping parameter names to lists of possible values:

.. code-block:: python

    search_space = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    }


Discrete Values
^^^^^^^^^^^^^^^

For parameters with a small set of distinct values:

.. code-block:: python

    search_space = {
        # Categorical choices
        "kernel": ["linear", "rbf", "poly"],

        # Boolean flags
        "fit_intercept": [True, False],

        # Specific integer values
        "n_neighbors": [3, 5, 7, 9, 11],
    }


Continuous Ranges
^^^^^^^^^^^^^^^^^

For parameters that vary continuously, use NumPy to create arrays:

.. code-block:: python

    import numpy as np

    search_space = {
        # Linear spacing for uniform ranges
        "momentum": np.linspace(0.5, 0.99, 50).tolist(),

        # Log spacing for parameters spanning orders of magnitude
        "learning_rate": np.logspace(-4, -1, 50).tolist(),

        # Integer range
        "hidden_size": list(range(32, 257, 32)),  # 32, 64, 96, ...
    }

.. tip::

    Convert NumPy arrays to lists with ``.tolist()`` for cleaner code,
    though Hyperactive accepts both formats.


Scale-Appropriate Spacing
-------------------------

Linear vs Logarithmic
^^^^^^^^^^^^^^^^^^^^^

The spacing between values should match how the parameter affects your objective:

**Linear spacing** — When changes have proportional effects:

.. code-block:: python

    # Dropout rate: 0.1 → 0.2 has similar effect as 0.5 → 0.6
    "dropout": np.linspace(0.0, 0.5, 11).tolist()

**Logarithmic spacing** — When the parameter spans orders of magnitude:

.. code-block:: python

    # Learning rate: 0.001 → 0.01 is as significant as 0.01 → 0.1
    "learning_rate": np.logspace(-4, -1, 20).tolist()

Common parameters that benefit from log spacing:

- Learning rates (``1e-5`` to ``1e-1``)
- Regularization strength (``1e-6`` to ``1e1``)
- Batch sizes (powers of 2: 16, 32, 64, 128, ...)


Choosing Granularity
--------------------

The number of values per parameter affects the total search space size:

.. code-block:: python

    # Fine granularity: 100 values per parameter
    # 3 parameters → 100^3 = 1,000,000 combinations
    "param_a": np.linspace(0, 1, 100).tolist(),
    "param_b": np.linspace(0, 1, 100).tolist(),
    "param_c": np.linspace(0, 1, 100).tolist(),

    # Coarse granularity: 10 values per parameter
    # 3 parameters → 10^3 = 1,000 combinations
    "param_a": np.linspace(0, 1, 10).tolist(),
    "param_b": np.linspace(0, 1, 10).tolist(),
    "param_c": np.linspace(0, 1, 10).tolist(),

**Guidelines:**

- Start coarse, refine after initial results
- Use finer granularity for sensitive parameters
- Use coarser granularity for less important parameters


Search Space Size Considerations
--------------------------------

Calculate your total search space size:

.. code-block:: python

    from functools import reduce
    import operator

    search_space = {
        "n_estimators": [10, 50, 100, 200],        # 4 values
        "max_depth": [3, 5, 10, None],              # 4 values
        "learning_rate": np.logspace(-3, 0, 20),   # 20 values
    }

    total_combinations = reduce(
        operator.mul,
        [len(v) for v in search_space.values()]
    )
    print(f"Total combinations: {total_combinations:,}")  # 320

**Recommendations by search space size:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Size
     - Recommended Approach
     - Optimizer Suggestions
   * - <100
     - Grid search (exhaustive)
     - ``GridSearch``
   * - 100–10,000
     - Random or local search
     - ``RandomSearch``, ``HillClimbing``
   * - 10,000–1,000,000
     - Smart sampling required
     - ``BayesianOptimizer``, ``TPE``
   * - >1,000,000
     - Reduce search space or use population methods
     - ``ParticleSwarmOptimizer``, ``EvolutionStrategy``


Handling Parameter Dependencies
-------------------------------

Sometimes parameters have dependencies. Handle these in your objective function:

.. code-block:: python

    def objective(params):
        # Constraint: min_samples_split >= min_samples_leaf
        if params["min_samples_split"] < params["min_samples_leaf"]:
            return -np.inf  # Invalid configuration

        # Constraint: kernel-specific parameters
        if params["kernel"] != "poly" and params["degree"] != 3:
            return -np.inf  # degree only relevant for poly kernel

        # Valid configuration — proceed with evaluation
        return evaluate_model(params)

.. note::

    Returning ``-np.inf`` effectively removes invalid combinations from consideration.


Common Search Space Patterns
----------------------------

Scikit-learn Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Random Forest
    rf_space = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    # Gradient Boosting
    gb_space = {
        "n_estimators": [50, 100, 200],
        "learning_rate": np.logspace(-3, 0, 10).tolist(),
        "max_depth": [3, 5, 7, 9],
        "subsample": np.linspace(0.6, 1.0, 5).tolist(),
    }

    # SVM
    svm_space = {
        "C": np.logspace(-2, 2, 10).tolist(),
        "gamma": np.logspace(-4, -1, 10).tolist(),
        "kernel": ["rbf", "poly", "sigmoid"],
    }


Neural Networks
^^^^^^^^^^^^^^^

.. code-block:: python

    nn_space = {
        "hidden_layers": [1, 2, 3],
        "hidden_size": [32, 64, 128, 256],
        "learning_rate": np.logspace(-4, -2, 20).tolist(),
        "dropout": np.linspace(0.0, 0.5, 6).tolist(),
        "batch_size": [16, 32, 64, 128],
        "activation": ["relu", "tanh", "elu"],
    }


Iterative Refinement Strategy
-----------------------------

A practical approach for finding optimal hyperparameters:

**Phase 1: Coarse Search**

.. code-block:: python

    # Wide ranges, few values
    coarse_space = {
        "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
        "hidden_size": [32, 128, 512],
        "dropout": [0.0, 0.25, 0.5],
    }

    optimizer = RandomSearch(
        search_space=coarse_space,
        n_iter=50,
        experiment=objective,
    )
    best = optimizer.solve()
    # Result: learning_rate=1e-3 works best

**Phase 2: Fine-tune Around Best Values**

.. code-block:: python

    # Narrow ranges around best from phase 1
    fine_space = {
        "learning_rate": np.logspace(-3.5, -2.5, 20).tolist(),  # Around 1e-3
        "hidden_size": list(range(96, 192, 16)),                # Around 128
        "dropout": np.linspace(0.2, 0.4, 10).tolist(),         # Around 0.25
    }

    optimizer = BayesianOptimizer(
        search_space=fine_space,
        n_iter=100,
        experiment=objective,
    )
    final_best = optimizer.solve()


Common Mistakes to Avoid
------------------------

**1. Overly Large Search Spaces**

.. code-block:: python

    # Bad: 1000 * 1000 * 1000 = 1 billion combinations
    bad_space = {
        "param_a": np.linspace(0, 1, 1000).tolist(),
        "param_b": np.linspace(0, 1, 1000).tolist(),
        "param_c": np.linspace(0, 1, 1000).tolist(),
    }

    # Better: 50 * 50 * 50 = 125,000 combinations
    better_space = {
        "param_a": np.linspace(0, 1, 50).tolist(),
        "param_b": np.linspace(0, 1, 50).tolist(),
        "param_c": np.linspace(0, 1, 50).tolist(),
    }

**2. Linear Spacing for Log-Scale Parameters**

.. code-block:: python

    # Bad: most values clustered at high end
    bad_lr = np.linspace(0.0001, 0.1, 20).tolist()
    # Values: 0.0001, 0.0053, 0.0106, ... (poor coverage of small values)

    # Good: even distribution across magnitudes
    good_lr = np.logspace(-4, -1, 20).tolist()
    # Values: 0.0001, 0.00016, 0.00025, ... 0.063, 0.1

**3. Missing Important Values**

.. code-block:: python

    # Bad: might miss optimal region entirely
    bad_space = {"max_depth": [2, 3, 4]}

    # Better: include None and reasonable range
    better_space = {"max_depth": [None, 3, 5, 10, 20, 50]}

**4. Ignoring Parameter Interactions**

Some parameters interact strongly. Consider them together:

.. code-block:: python

    # Learning rate and batch size often interact
    # Higher batch sizes often need higher learning rates
    search_space = {
        "batch_size": [16, 32, 64, 128],
        "learning_rate": np.logspace(-4, -1, 20).tolist(),
    }
    # The optimizer will explore combinations to find the best pairing
