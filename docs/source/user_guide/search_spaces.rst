.. _user_guide_search_spaces:

=============
Search Spaces
=============

The search space defines all configurations the optimizer can explore. Too narrow and
you miss good solutions; too broad and the optimizer wastes iterations. In Hyperactive,
search spaces are dictionaries mapping parameter names to lists of possible values.

----

What is a Search Space?
-----------------------

A search space defines all possible parameter combinations the optimizer can explore.
It's simply a dictionary mapping parameter names to lists of values.

.. raw:: html

   <div class="theme-aware-diagram">
      <img src="../_static/diagrams/search_space_concept_light.svg"
           alt="Search space concept: parameters expand into combinations"
           class="only-light" />
      <img src="../_static/diagrams/search_space_concept_dark.svg"
           alt="Search space concept: parameters expand into combinations"
           class="only-dark" />
   </div>

|

.. code-block:: python

   search_space = {
       "learning_rate": [0.001, 0.01, 0.1],      # 3 values
       "n_estimators": [50, 100, 200, 500],      # 4 values
       "max_depth": [3, 5, 10, None],            # 4 values
   }
   # Total: 3 × 4 × 4 = 48 combinations

----

Parameter Types
---------------

Hyperactive supports any value that can be stored in a Python list:

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Categorical
      :class-card: sd-border-primary

      Discrete choices like strings or objects.

      .. code-block:: python

         "kernel": ["linear", "rbf", "poly"]
         "optimizer": [Adam, SGD, RMSprop]

   .. grid-item-card:: Integer
      :class-card: sd-border-success

      Discrete numeric values.

      .. code-block:: python

         "n_estimators": [50, 100, 200]
         "hidden_size": list(range(32, 257, 32))

   .. grid-item-card:: Continuous
      :class-card: sd-border-warning

      Float values (discretized into steps).

      .. code-block:: python

         "dropout": np.linspace(0, 0.5, 11)
         "learning_rate": np.logspace(-4, -1, 20)

----

Linear vs Logarithmic Spacing
-----------------------------

The spacing between values matters. Choose based on how the parameter affects your objective.

.. tab-set::

   .. tab-item:: Linear Spacing

      Use when equal differences have equal effects.

      .. code-block:: python

         # Dropout: 0.1 → 0.2 has similar effect as 0.4 → 0.5
         "dropout": np.linspace(0.0, 0.5, 11).tolist()
         # [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

   .. tab-item:: Log Spacing

      Use when the parameter spans orders of magnitude.

      .. code-block:: python

         # Learning rate: 0.001 → 0.01 is as significant as 0.01 → 0.1
         "learning_rate": np.logspace(-4, -1, 10).tolist()
         # [0.0001, 0.00028, 0.00077, 0.00215, 0.00599, 0.01668, 0.04642, 0.1]

**Common parameters that benefit from log spacing:**

- Learning rates (``1e-5`` to ``1e-1``)
- Regularization strength (``1e-6`` to ``1e1``)
- Batch sizes (powers of 2)

----

Granularity and Size
--------------------

More values per parameter means more combinations to explore.

.. admonition:: The Multiplication Effect
   :class: important

   With 3 parameters, each having 10 values: 10 × 10 × 10 = **1,000 combinations**

   With 3 parameters, each having 100 values: 100 × 100 × 100 = **1,000,000 combinations**

Calculate your search space size:

.. code-block:: python

   from functools import reduce
   import operator

   total = reduce(operator.mul, [len(v) for v in search_space.values()])
   print(f"Total combinations: {total:,}")

**Size recommendations:**

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Size
     - Approach
     - Recommended Optimizers
   * - < 1,000
     - Exhaustive or random search
     - ``GridSearch``, ``RandomSearch``
   * - 1,000 - 100,000
     - Smart sampling
     - ``BayesianOptimizer``, ``TPE``
   * - 100,000 - 10M
     - Population or local methods
     - ``ParticleSwarm``, ``HillClimbing``
   * - > 10M
     - Reduce space or use iterative refinement
     - Start coarse, then refine

.. tip::

   Start with a coarse search space (fewer values per parameter), then refine around
   the best region found.

----

Common Patterns
---------------

Ready-to-use search spaces for common ML models:

.. tab-set::

   .. tab-item:: Random Forest

      .. code-block:: python

         rf_space = {
             "n_estimators": [50, 100, 200, 500],
             "max_depth": [None, 5, 10, 20, 30],
             "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4],
             "max_features": ["sqrt", "log2", None],
         }

   .. tab-item:: Gradient Boosting

      .. code-block:: python

         import numpy as np

         gb_space = {
             "n_estimators": [50, 100, 200, 500],
             "learning_rate": np.logspace(-3, 0, 10).tolist(),
             "max_depth": [3, 5, 7, 9],
             "subsample": np.linspace(0.6, 1.0, 5).tolist(),
         }

   .. tab-item:: SVM

      .. code-block:: python

         import numpy as np

         svm_space = {
             "C": np.logspace(-2, 2, 10).tolist(),
             "gamma": np.logspace(-4, -1, 10).tolist(),
             "kernel": ["rbf", "poly", "sigmoid"],
         }

   .. tab-item:: Neural Network

      .. code-block:: python

         import numpy as np

         nn_space = {
             "hidden_layers": [1, 2, 3],
             "hidden_size": [32, 64, 128, 256],
             "learning_rate": np.logspace(-4, -2, 20).tolist(),
             "dropout": np.linspace(0.0, 0.5, 6).tolist(),
             "batch_size": [16, 32, 64, 128],
             "activation": ["relu", "tanh", "elu"],
         }

----

Parameter Dependencies
----------------------

Sometimes parameters have constraints or dependencies. Handle these in your experiment function:

.. code-block:: python

   import numpy as np

   def experiment(params):
       # Constraint: min_samples_split >= min_samples_leaf
       if params["min_samples_split"] < params["min_samples_leaf"]:
           return -np.inf  # Invalid configuration

       # Constraint: degree only relevant for poly kernel
       if params["kernel"] != "poly" and params["degree"] != 3:
           return -np.inf

       # Valid configuration
       return evaluate_model(params)

.. note::

   Returning ``-np.inf`` effectively removes invalid combinations from consideration.
   The optimizer will learn to avoid these regions.

----

Iterative Refinement
--------------------

A practical two-phase approach for finding optimal hyperparameters:

**Phase 1: Coarse Search**

Explore wide ranges with few values to find promising regions.

.. code-block:: python

   coarse_space = {
       "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
       "hidden_size": [32, 128, 512],
       "dropout": [0.0, 0.25, 0.5],
   }

   optimizer = RandomSearch(coarse_space, n_iter=50, experiment=objective)
   best = optimizer.solve()
   # Result: learning_rate=1e-3 region looks promising

**Phase 2: Fine Search**

Narrow in on the best region with more granularity.

.. code-block:: python

   fine_space = {
       "learning_rate": np.logspace(-3.5, -2.5, 20).tolist(),  # Around 1e-3
       "hidden_size": list(range(96, 192, 16)),                 # Around 128
       "dropout": np.linspace(0.2, 0.4, 10).tolist(),          # Around 0.25
   }

   optimizer = BayesianOptimizer(fine_space, n_iter=100, experiment=objective)
   final_best = optimizer.solve()

----

Common Mistakes
---------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Overly Large Spaces

      .. code-block:: python

         # Bad: 1000³ = 1 billion combinations
         "param": np.linspace(0, 1, 1000)

         # Better: 50³ = 125,000 combinations
         "param": np.linspace(0, 1, 50)

   .. grid-item-card:: Wrong Spacing

      .. code-block:: python

         # Bad: poor coverage of small values
         "lr": np.linspace(0.0001, 0.1, 20)

         # Good: even coverage across magnitudes
         "lr": np.logspace(-4, -1, 20)

   .. grid-item-card:: Missing Values

      .. code-block:: python

         # Bad: might miss optimal region
         "max_depth": [2, 3, 4]

         # Better: include None and wider range
         "max_depth": [None, 3, 5, 10, 20, 50]

   .. grid-item-card:: Too Fine Initially

      .. code-block:: python

         # Bad for initial search
         "lr": np.logspace(-4, -1, 100)

         # Better: start coarse, refine later
         "lr": np.logspace(-4, -1, 10)

----

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Parameter Type
     - Example
     - When to Use
   * - Categorical
     - ``["rbf", "linear", "poly"]``
     - Distinct choices
   * - Integer range
     - ``list(range(10, 101, 10))``
     - Discrete numeric parameters
   * - Linear float
     - ``np.linspace(0, 1, 20).tolist()``
     - Uniform parameters (dropout, momentum)
   * - Log float
     - ``np.logspace(-4, -1, 20).tolist()``
     - Multi-magnitude parameters (learning rate)
   * - Boolean
     - ``[True, False]``
     - Toggle features
