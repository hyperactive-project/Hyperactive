.. _examples:

========
Examples
========

This page provides a collection of examples demonstrating Hyperactive's capabilities.
All examples are available in the
`examples directory <https://github.com/SimonBlanke/Hyperactive/tree/master/examples>`_
on GitHub.


Example Gallery
---------------

Hyperactive includes examples for various optimization algorithms and use cases.
You can run any example directly:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/SimonBlanke/Hyperactive.git
    cd Hyperactive/examples

    # Run an example
    python gfo/hill_climbing_example.py


Basic Examples
--------------

Custom Function Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest use case: optimizing a mathematical function.

.. code-block:: python

    import numpy as np
    from hyperactive.opt.gfo import HillClimbing

    def objective(params):
        x = params["x"]
        y = params["y"]
        return -(x**2 + y**2)  # Maximize (minimize the parabola)

    search_space = {
        "x": np.arange(-5, 5, 0.1),
        "y": np.arange(-5, 5, 0.1),
    }

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )
    best_params = optimizer.solve()
    print(f"Best parameters: {best_params}")


Scikit-learn Model Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^

Hyperparameter optimization for machine learning models.

.. code-block:: python

    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from hyperactive.opt.gfo import HillClimbing

    # Load data
    X, y = load_wine(return_X_y=True)

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        X=X, y=y, cv=3,
    )

    # Define search space
    search_space = {
        "n_estimators": list(range(10, 201)),
        "max_depth": list(range(1, 21)),
        "min_samples_split": list(range(2, 21)),
        "min_samples_leaf": list(range(1, 11)),
    }

    # Optimize
    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=40,
        random_state=42,
        experiment=experiment,
    )
    best_params = optimizer.solve()


Gradient-Free Optimizer Examples
--------------------------------

Local Search Algorithms
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Hill Climbing
     - `hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/hill_climbing_example.py>`_
   * - Repulsing Hill Climbing
     - `repulsing_hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/repulsing_hill_climbing_example.py>`_
   * - Simulated Annealing
     - `simulated_annealing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/simulated_annealing_example.py>`_
   * - Downhill Simplex
     - `downhill_simplex_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/downhill_simplex_example.py>`_


Global Search Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Random Search
     - `random_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/random_search_example.py>`_
   * - Grid Search
     - `grid_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/grid_search_example.py>`_
   * - Random Restart Hill Climbing
     - `random_restart_hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/random_restart_hill_climbing_example.py>`_
   * - Stochastic Hill Climbing
     - `stochastic_hill_climbing_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/stochastic_hill_climbing_example.py>`_
   * - Powell's Method
     - `powells_method_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/powells_method_example.py>`_
   * - Pattern Search
     - `pattern_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/pattern_search_example.py>`_


Population-Based Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Particle Swarm
     - `particle_swarm_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/particle_swarm_example.py>`_
   * - Genetic Algorithm
     - `genetic_algorithm_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/genetic_algorithm_example.py>`_
   * - Evolution Strategy
     - `evolution_strategy_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/evolution_strategy_example.py>`_
   * - Differential Evolution
     - `differential_evolution_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/differential_evolution_example.py>`_
   * - Parallel Tempering
     - `parallel_tempering_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/parallel_tempering_example.py>`_
   * - Spiral Optimization
     - `spiral_optimization_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/spiral_optimization_example.py>`_


Sequential Model-Based Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - Bayesian Optimization
     - `bayesian_optimization_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/bayesian_optimization_example.py>`_
   * - Tree-Parzen Estimators
     - `tree_structured_parzen_estimators_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/tree_structured_parzen_estimators_example.py>`_
   * - Forest Optimizer
     - `forest_optimizer_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/forest_optimizer_example.py>`_
   * - Lipschitz Optimizer
     - `lipschitz_optimizer_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/lipschitz_optimizer_example.py>`_
   * - DIRECT Algorithm
     - `direct_algorithm_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/gfo/direct_algorithm_example.py>`_


Optuna Backend Examples
-----------------------

Examples using Optuna's optimization algorithms:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - Example
   * - TPE Optimizer
     - `tpe_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/tpe_sampler_example.py>`_
   * - CMA-ES
     - `cmaes_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/cmaes_sampler_example.py>`_
   * - Gaussian Process
     - `gp_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/gp_sampler_example.py>`_
   * - NSGA-II
     - `nsga_ii_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/nsga_ii_sampler_example.py>`_
   * - NSGA-III
     - `nsga_iii_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/nsga_iii_sampler_example.py>`_
   * - QMC
     - `qmc_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/qmc_sampler_example.py>`_
   * - Random
     - `random_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/random_sampler_example.py>`_
   * - Grid
     - `grid_sampler_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/optuna/grid_sampler_example.py>`_


Integration Examples
--------------------

Scikit-learn Integration
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Example
   * - Classification with OptCV
     - `sklearn_classif_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/integrations/sklearn_classif_example.py>`_
   * - Grid Search
     - `grid_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/sklearn/grid_search_example.py>`_
   * - Random Search
     - `random_search_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/sklearn/random_search_example.py>`_


Sktime Integration
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Example
   * - Time Series Forecasting
     - `sktime_forecasting_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/integrations/sktime_forecasting_example.py>`_
   * - Time Series Classification
     - `sktime_tsc_example.py <https://github.com/SimonBlanke/Hyperactive/blob/master/examples/integrations/sktime_tsc_example.py>`_


Advanced Examples
-----------------

Warm Starting Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Start optimization from known good points:

.. code-block:: python

    from hyperactive.opt.gfo import HillClimbing

    # Previous best parameters
    warm_start_points = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
    ]

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=40,
        experiment=experiment,
        initialize={"warm_start": warm_start_points},
    )
    best_params = optimizer.solve()


Comparing Optimizers
^^^^^^^^^^^^^^^^^^^^

Compare different optimization strategies:

.. code-block:: python

    from hyperactive.opt.gfo import (
        HillClimbing,
        RandomSearch,
        BayesianOptimizer,
        ParticleSwarmOptimizer,
    )

    optimizers = {
        "HillClimbing": HillClimbing,
        "RandomSearch": RandomSearch,
        "Bayesian": BayesianOptimizer,
        "ParticleSwarm": ParticleSwarmOptimizer,
    }

    results = {}
    for name, OptClass in optimizers.items():
        optimizer = OptClass(
            search_space=search_space,
            n_iter=50,
            experiment=experiment,
            random_state=42,
        )
        best = optimizer.solve()
        score, _ = experiment.score(best)
        results[name] = {"params": best, "score": score}
        print(f"{name}: score={score:.4f}")


Interactive Tutorial
--------------------

For a comprehensive interactive tutorial, see the
`Hyperactive Tutorial Notebook <https://nbviewer.org/github/SimonBlanke/hyperactive-tutorial/blob/main/notebooks/hyperactive_tutorial.ipynb>`_.

This Jupyter notebook covers:

- Basic optimization concepts
- All optimizer categories
- Real-world ML examples
- Best practices and tips
