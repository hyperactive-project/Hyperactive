.. _user_guide_introduction:

============
Introduction
============

This page introduces Hyperactive's core concepts: optimizers, experiments, and search spaces.
Understanding these concepts will help you use Hyperactive effectively for any optimization task.


Core Concepts
-------------

Hyperactive is built around three key concepts:

1. **Experiments** — Define *what* to optimize (the objective function)
2. **Optimizers** — Define *how* to optimize (the search algorithm)
3. **Search Spaces** — Define *where* to search (the parameter ranges)


Experiments: What to Optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An **experiment** represents your optimization problem. It takes parameters as input
and returns a score that Hyperactive will maximize.

The simplest experiment is a Python function:

.. code-block:: python

    def objective(params):
        x = params["x"]
        y = params["y"]
        # Return a score to maximize
        return -(x**2 + y**2)

For machine learning, Hyperactive provides built-in experiments:

.. code-block:: python

    from hyperactive.experiment.integrations import SklearnCvExperiment

    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(),
        X=X_train,
        y=y_train,
        cv=5,
    )

See :ref:`user_guide_experiments` for more details.


Optimizers: How to Optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An **optimizer** is the algorithm that explores the search space to find the best parameters.
Hyperactive provides 20+ optimizers in different categories:

.. code-block:: python

    from hyperactive.opt.gfo import (
        HillClimbing,           # Local search
        RandomSearch,           # Global search
        BayesianOptimizer,      # Sequential model-based
        ParticleSwarmOptimizer, # Population-based
    )

Each optimizer has different characteristics:

- **Local search** (HillClimbing, SimulatedAnnealing): Fast, may get stuck in local optima
- **Global search** (RandomSearch, GridSearch): Thorough exploration, slower
- **Population methods** (GeneticAlgorithm, ParticleSwarm): Good for complex landscapes
- **Sequential methods** (BayesianOptimizer, TPE): Smart exploration, best for expensive evaluations

See :ref:`user_guide_optimizers` for a complete guide.


Search Spaces: Where to Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **search space** defines the possible values for each parameter.
Use dictionaries with lists or NumPy arrays:

.. code-block:: python

    import numpy as np

    search_space = {
        # Discrete integer values
        "n_estimators": list(range(10, 200, 10)),

        # Continuous values (discretized)
        "learning_rate": np.logspace(-4, 0, 20),

        # Categorical values
        "kernel": ["linear", "rbf", "poly"],
    }

.. tip::

    Keep search spaces reasonably sized. Very large spaces (>10^8 combinations)
    can cause memory issues with some optimizers.


Basic Workflow
--------------

Here's the complete workflow for using Hyperactive:

Step 1: Define Your Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Either as a function or using built-in experiment classes:

.. code-block:: python

    # Option A: Custom function
    def my_objective(params):
        # Your evaluation logic here
        return score

    # Option B: Built-in sklearn experiment
    from hyperactive.experiment.integrations import SklearnCvExperiment

    experiment = SklearnCvExperiment(
        estimator=YourEstimator(),
        X=X, y=y, cv=5,
    )


Step 2: Define the Search Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    search_space = {
        "param1": [1, 2, 3, 4, 5],
        "param2": np.linspace(0.1, 1.0, 10),
        "param3": ["option_a", "option_b"],
    }


Step 3: Choose an Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from hyperactive.opt.gfo import HillClimbing

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,           # Number of iterations
        experiment=experiment,
        random_state=42,      # For reproducibility
    )


Step 4: Run the Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    best_params = optimizer.solve()
    print(f"Best parameters: {best_params}")


Common Optimizer Parameters
---------------------------

Most optimizers share these parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``search_space``
     - dict
     - Maps parameter names to possible values
   * - ``n_iter``
     - int
     - Number of optimization iterations
   * - ``experiment``
     - callable
     - The objective function or experiment object
   * - ``random_state``
     - int
     - Seed for reproducibility
   * - ``initialize``
     - dict
     - Control initial population (warm starts, etc.)


Warm Starting
^^^^^^^^^^^^^

You can provide starting points for optimization:

.. code-block:: python

    warm_start = [
        {"n_estimators": 100, "max_depth": 10},  # Start from known good point
    ]

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=50,
        experiment=experiment,
        initialize={"warm_start": warm_start},
    )


Tips for Effective Optimization
-------------------------------

1. **Start simple**: Begin with ``HillClimbing`` or ``RandomSearch`` to establish baselines.

2. **Right-size your search space**: Large spaces need more iterations. Consider using
   ``np.logspace`` for parameters that span orders of magnitude.

3. **Use appropriate iterations**: More iterations = better exploration, but longer runtime.
   A good rule of thumb: at least 10x the number of parameters.

4. **Set random_state**: For reproducible results, always set a random seed.

5. **Consider your budget**: For expensive evaluations (training large models),
   use smart optimizers like ``BayesianOptimizer`` that learn from previous evaluations.
