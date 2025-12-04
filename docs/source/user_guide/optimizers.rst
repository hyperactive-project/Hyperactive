.. _user_guide_optimizers:

==========
Optimizers
==========

Optimizers define *how* Hyperactive explores the search space to find optimal parameters.
This guide helps you choose the right optimizer for your problem and configure it effectively.


Choosing an Optimizer
---------------------

The best optimizer depends on your problem characteristics:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Scenario
     - Recommended Optimizers
     - Why
   * - Quick baseline
     - ``HillClimbing``, ``RandomSearch``
     - Fast, simple, good for initial exploration
   * - Expensive evaluations
     - ``BayesianOptimizer``, ``TPEOptimizer``
     - Learn from past evaluations, minimize function calls
   * - Large search space
     - ``RandomSearch``, ``ParticleSwarmOptimizer``
     - Good global coverage
   * - Multi-modal landscape
     - ``GeneticAlgorithm``, ``DifferentialEvolution``
     - Population-based, avoid local optima
   * - Small search space
     - ``GridSearch``
     - Exhaustive coverage when feasible


Optimizer Categories
--------------------

Hyperactive organizes optimizers into categories based on their search strategies.


Local Search
^^^^^^^^^^^^

Local search optimizers explore the neighborhood of the current best solution.
They're fast but may get stuck in local optima.

**Hill Climbing**

The simplest local search: always move to a better neighbor.

.. code-block:: python

    from hyperactive.opt.gfo import HillClimbing

    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Simulated Annealing**

Like hill climbing, but sometimes accepts worse solutions to escape local optima.
The "temperature" controls exploration vs exploitation.

.. code-block:: python

    from hyperactive.opt.gfo import SimulatedAnnealing

    optimizer = SimulatedAnnealing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Repulsing Hill Climbing**

Remembers visited regions and avoids them, encouraging broader exploration.

.. code-block:: python

    from hyperactive.opt.gfo import RepulsingHillClimbing

    optimizer = RepulsingHillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Downhill Simplex (Nelder-Mead)**

Uses a simplex of points to navigate the search space. Good for continuous problems.

.. code-block:: python

    from hyperactive.opt.gfo import DownhillSimplexOptimizer

    optimizer = DownhillSimplexOptimizer(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )


Global Search
^^^^^^^^^^^^^

Global search optimizers explore the entire search space more thoroughly.

**Random Search**

Samples random points from the search space. Simple but surprisingly effective baseline.

.. code-block:: python

    from hyperactive.opt.gfo import RandomSearch

    optimizer = RandomSearch(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Grid Search**

Evaluates all combinations systematically. Only practical for small search spaces.

.. code-block:: python

    from hyperactive.opt.gfo import GridSearch

    optimizer = GridSearch(
        search_space=search_space,
        experiment=objective,
    )

**Random Restart Hill Climbing**

Runs hill climbing from multiple random starting points.

.. code-block:: python

    from hyperactive.opt.gfo import RandomRestartHillClimbing

    optimizer = RandomRestartHillClimbing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Powell's Method** and **Pattern Search**

Classical derivative-free optimization methods.

.. code-block:: python

    from hyperactive.opt.gfo import PowellsMethod, PatternSearch


Population Methods
^^^^^^^^^^^^^^^^^^

Population-based optimizers maintain multiple candidate solutions that evolve together.
They're excellent for complex, multi-modal optimization landscapes.

**Particle Swarm Optimization**

Particles "fly" through the search space, influenced by their own best position
and the swarm's best position.

.. code-block:: python

    from hyperactive.opt.gfo import ParticleSwarmOptimizer

    optimizer = ParticleSwarmOptimizer(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Genetic Algorithm**

Evolves a population using selection, crossover, and mutation inspired by biology.

.. code-block:: python

    from hyperactive.opt.gfo import GeneticAlgorithm

    optimizer = GeneticAlgorithm(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
    )

**Evolution Strategy**

Similar to genetic algorithms but focused on real-valued optimization.

.. code-block:: python

    from hyperactive.opt.gfo import EvolutionStrategy

**Differential Evolution**

Uses vector differences to guide mutation. Excellent for continuous optimization.

.. code-block:: python

    from hyperactive.opt.gfo import DifferentialEvolution

**Parallel Tempering**

Runs multiple chains at different "temperatures" and exchanges information between them.

.. code-block:: python

    from hyperactive.opt.gfo import ParallelTempering

**Spiral Optimization**

Particles spiral toward the best solution found so far.

.. code-block:: python

    from hyperactive.opt.gfo import SpiralOptimization


Sequential Model-Based (Bayesian)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These optimizers build a model of the objective function and use it to decide
where to sample next. Best for expensive evaluations.

**Bayesian Optimization**

Uses Gaussian Process regression to model the objective and acquisition functions
to balance exploration and exploitation.

.. code-block:: python

    from hyperactive.opt.gfo import BayesianOptimizer

    optimizer = BayesianOptimizer(
        search_space=search_space,
        n_iter=50,
        experiment=objective,
    )

**Tree-Structured Parzen Estimators (TPE)**

Models the distribution of good and bad parameters separately.

.. code-block:: python

    from hyperactive.opt.gfo import TreeStructuredParzenEstimators

**Forest Optimizer**

Uses Random Forest to model the objective function.

.. code-block:: python

    from hyperactive.opt.gfo import ForestOptimizer

**Lipschitz Optimization** and **DIRECT Algorithm**

Use Lipschitz continuity assumptions to guide the search.

.. code-block:: python

    from hyperactive.opt.gfo import LipschitzOptimizer, DirectAlgorithm


Optuna Backend
^^^^^^^^^^^^^^

Hyperactive provides wrappers for Optuna's powerful samplers:

.. code-block:: python

    from hyperactive.opt.optuna import (
        TPEOptimizer,       # Tree-Parzen Estimators
        CmaEsOptimizer,     # CMA-ES evolution strategy
        GPOptimizer,        # Gaussian Process
        NSGAIIOptimizer,    # Multi-objective (NSGA-II)
        NSGAIIIOptimizer,   # Multi-objective (NSGA-III)
        QMCOptimizer,       # Quasi-Monte Carlo
        RandomOptimizer,    # Random sampling
        GridOptimizer,      # Grid search
    )

Example with Optuna TPE:

.. code-block:: python

    from hyperactive.opt.optuna import TPEOptimizer

    optimizer = TPEOptimizer(
        search_space=search_space,
        n_iter=50,
        experiment=objective,
    )


Optimizer Configuration
-----------------------

Common Parameters
^^^^^^^^^^^^^^^^^

All optimizers accept these parameters:

.. code-block:: python

    optimizer = SomeOptimizer(
        search_space=search_space,  # Required: parameter ranges
        n_iter=100,                  # Required: number of iterations
        experiment=objective,        # Required: objective function
        random_state=42,             # Optional: for reproducibility
        initialize={                 # Optional: initialization settings
            "warm_start": [...],     # Starting points
            "grid": 4,               # Grid initialization points
            "random": 2,             # Random initialization points
            "vertices": 4,           # Vertex initialization points
        },
    )


Initialization Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

Control how the optimizer initializes its search:

.. code-block:: python

    # Start from known good points
    optimizer = HillClimbing(
        search_space=search_space,
        n_iter=50,
        experiment=objective,
        initialize={
            "warm_start": [
                {"param1": 10, "param2": 0.5},
                {"param1": 20, "param2": 0.3},
            ]
        },
    )

    # Mix of initialization strategies
    optimizer = ParticleSwarmOptimizer(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
        initialize={
            "grid": 4,      # 4 points on a grid
            "random": 6,    # 6 random points
            "vertices": 4,  # 4 corner points
        },
    )


Algorithm-Specific Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many optimizers have additional parameters. Check the :ref:`api_reference` for details.

Example with Simulated Annealing:

.. code-block:: python

    from hyperactive.opt.gfo import SimulatedAnnealing

    optimizer = SimulatedAnnealing(
        search_space=search_space,
        n_iter=100,
        experiment=objective,
        # Algorithm-specific parameters
        # (check API reference for available options)
    )


Performance Tips
----------------

1. **Start with baselines**: Always run ``RandomSearch`` first to establish
   a baseline and understand your objective landscape.

2. **Match iterations to complexity**: Complex optimizers (Bayesian, population-based)
   need more iterations to show their advantages.

3. **Consider evaluation cost**: For cheap evaluations, simple optimizers work well.
   For expensive ones, use model-based approaches.

4. **Use warm starts**: If you have prior knowledge, warm start can significantly
   speed up optimization.

5. **Set random seeds**: For reproducible results, always set ``random_state``.
