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

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:hill_climbing]
   :end-before: # [end:hill_climbing]

**Simulated Annealing**

Like hill climbing, but sometimes accepts worse solutions to escape local optima.
The "temperature" controls exploration vs exploitation.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:simulated_annealing]
   :end-before: # [end:simulated_annealing]

**Repulsing Hill Climbing**

Remembers visited regions and avoids them, encouraging broader exploration.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:repulsing_hill_climbing]
   :end-before: # [end:repulsing_hill_climbing]

**Stochastic Hill Climbing**

Hill climbing with a probability of accepting worse solutions. The ``p_accept``
parameter controls exploration — higher values make it more likely to accept
non-improving moves, helping escape local optima.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:stochastic_hill_climbing]
   :end-before: # [end:stochastic_hill_climbing]

**Downhill Simplex (Nelder-Mead)**

Uses a simplex of points to navigate the search space. Good for continuous problems.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:downhill_simplex]
   :end-before: # [end:downhill_simplex]


Global Search
^^^^^^^^^^^^^

Global search optimizers explore the entire search space more thoroughly.

**Random Search**

Samples random points from the search space. Simple but surprisingly effective baseline.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:random_search]
   :end-before: # [end:random_search]

**Grid Search**

Evaluates all combinations systematically. Only practical for small search spaces.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:grid_search]
   :end-before: # [end:grid_search]

**Random Restart Hill Climbing**

Runs hill climbing from multiple random starting points.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:random_restart_hill_climbing]
   :end-before: # [end:random_restart_hill_climbing]

**Powell's Method** and **Pattern Search**

Classical derivative-free optimization methods.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:powells_pattern]
   :end-before: # [end:powells_pattern]


Population Methods
^^^^^^^^^^^^^^^^^^

Population-based optimizers maintain multiple candidate solutions that evolve together.
They're excellent for complex, multi-modal optimization landscapes.

**Particle Swarm Optimization**

Particles "fly" through the search space, influenced by their own best position
and the swarm's best position.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:particle_swarm]
   :end-before: # [end:particle_swarm]

**Genetic Algorithm**

Evolves a population using selection, crossover, and mutation inspired by biology.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:genetic_algorithm]
   :end-before: # [end:genetic_algorithm]

**Evolution Strategy**

Similar to genetic algorithms but focused on real-valued optimization.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:evolution_strategy]
   :end-before: # [end:evolution_strategy]

**Differential Evolution**

Uses vector differences to guide mutation. Excellent for continuous optimization.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:differential_evolution]
   :end-before: # [end:differential_evolution]

**Parallel Tempering**

Runs multiple chains at different "temperatures" and exchanges information between them.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:parallel_tempering]
   :end-before: # [end:parallel_tempering]

**Spiral Optimization**

Particles spiral toward the best solution found so far.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:spiral_optimization]
   :end-before: # [end:spiral_optimization]


Sequential Model-Based (Bayesian)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These optimizers build a model of the objective function and use it to decide
where to sample next. Best for expensive evaluations.

**Bayesian Optimization**

Uses Gaussian Process regression to model the objective and acquisition functions
to balance exploration and exploitation.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:bayesian_optimizer]
   :end-before: # [end:bayesian_optimizer]

**Tree-Structured Parzen Estimators (TPE)**

Models the distribution of good and bad parameters separately.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:tpe]
   :end-before: # [end:tpe]

**Forest Optimizer**

Uses Random Forest to model the objective function.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:forest_optimizer]
   :end-before: # [end:forest_optimizer]

**Lipschitz Optimization** and **DIRECT Algorithm**

Use Lipschitz continuity assumptions to guide the search.

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:lipschitz_direct]
   :end-before: # [end:lipschitz_direct]


Optuna Backend
^^^^^^^^^^^^^^

Hyperactive provides wrappers for Optuna's powerful samplers:

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:optuna_imports]
   :end-before: # [end:optuna_imports]

Example with Optuna TPE:

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:optuna_tpe]
   :end-before: # [end:optuna_tpe]


Optimizer Configuration
-----------------------

Common Parameters
^^^^^^^^^^^^^^^^^

All optimizers accept these parameters:

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:common_parameters]
   :end-before: # [end:common_parameters]


Initialization Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

Control how the optimizer initializes its search:

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:warm_start_example]
   :end-before: # [end:warm_start_example]

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:initialization_strategies]
   :end-before: # [end:initialization_strategies]


Algorithm-Specific Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many optimizers have additional parameters. Check the :ref:`api_reference` for details.

Example with Simulated Annealing:

.. literalinclude:: ../_snippets/user_guide/optimizers.py
   :language: python
   :start-after: # [start:simulated_annealing_config]
   :end-before: # [end:simulated_annealing_config]


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
