.. _user_guide_optimizers:

==========
Optimizers
==========

Hyperactive provides 34 algorithms across 5 categories and 4 backends.
Optimizers navigate the search space to find optimal parameters. Each implements a
different strategy for balancing exploration (trying diverse regions) and exploitation
(refining promising solutions). Local search methods like Hill Climbing work well for
smooth landscapes. Population-based methods handle multiple local optima. Model-based
methods like Bayesian Optimization minimize evaluations for expensive objective functions.

----

Algorithm Landscape
-------------------

.. raw:: html

   <div class="theme-aware-diagram">
      <img src="../../_static/diagrams/optimizer_taxonomy_light.svg"
           alt="Hyperactive optimizer taxonomy showing 31 algorithms across GFO, Optuna, and sklearn backends"
           class="only-light" />
      <img src="../../_static/diagrams/optimizer_taxonomy_dark.svg"
           alt="Hyperactive optimizer taxonomy showing 31 algorithms across GFO, Optuna, and sklearn backends"
           class="only-dark" />
   </div>

|

Quick Selection Guide
---------------------

Not sure which optimizer to use? Start here:

.. grid:: 2 2 4 4
   :gutter: 3

   .. grid-item-card:: Quick baseline
      :class-card: sd-bg-light

      **Use:** ``RandomSearch`` or ``HillClimbing``

      Fast, simple, good starting point for any problem.

   .. grid-item-card:: Expensive evaluations
      :class-card: sd-bg-light

      **Use:** ``BayesianOptimizer`` or ``TPEOptimizer``

      Learn from past evaluations, minimize function calls.

   .. grid-item-card:: Many local optima
      :class-card: sd-bg-light

      **Use:** ``GeneticAlgorithm`` or ``DifferentialEvolution``

      Population-based methods escape local traps.

   .. grid-item-card:: Small search space
      :class-card: sd-bg-light

      **Use:** ``GridSearch``

      Exhaustive coverage when feasible (<1000 combinations).

.. tip::

   Always run ``RandomSearch`` first to establish a baseline and understand your
   objective landscape before trying more sophisticated algorithms.

----

Algorithm Categories
--------------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Local Search
      :link: local_search
      :link-type: doc
      :class-card: sd-border-danger

      **5 algorithms**
      ^^^
      Explore neighborhoods of current solutions. Fast but may get stuck in local optima.

      *HillClimbing, SimulatedAnnealing, StochasticHillClimbing, RepulsingHillClimbing, DownhillSimplex*

   .. grid-item-card:: Global Search
      :link: global_search
      :link-type: doc
      :class-card: sd-border-warning

      **4 algorithms**
      ^^^
      Explore the entire search space systematically or randomly.

      *RandomSearch, GridSearch, RandomRestartHillClimbing, Powell's Method*

   .. grid-item-card:: Population-Based
      :link: population_based
      :link-type: doc
      :class-card: sd-border-success

      **6 algorithms**
      ^^^
      Evolve multiple candidate solutions together. Excellent for complex landscapes.

      *ParticleSwarm, GeneticAlgorithm, EvolutionStrategy, DifferentialEvolution, ParallelTempering, SpiralOptimization*

   .. grid-item-card:: Model-Based (SMBO)
      :link: sequential_model_based
      :link-type: doc
      :class-card: sd-border-primary

      **5 algorithms**
      ^^^
      Build surrogate models to guide search. Best for expensive evaluations.

      *BayesianOptimizer, TPE, ForestOptimizer, Lipschitz, DIRECT*

   .. grid-item-card:: Optuna Backend
      :link: optuna
      :link-type: doc
      :class-card: sd-border-info

      **8 algorithms**
      ^^^
      Wrappers for Optuna's powerful samplers with Hyperactive's interface.

      *TPEOptimizer, CmaEsOptimizer, GPOptimizer, NSGAIIOptimizer, and more*

   .. grid-item-card:: SMAC Backend
      :link: smac
      :link-type: doc
      :class-card: sd-border-secondary

      **3 algorithms**
      ^^^
      State-of-the-art Bayesian optimization from the AutoML community.

      *SmacRandomForest, SmacGaussianProcess, SmacRandomSearch*

----

Scenario Reference
------------------

Detailed recommendations based on problem characteristics:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Scenario
     - Recommended Optimizers
     - Why
   * - Quick baseline
     - ``HillClimbing``, ``RandomSearch``
     - Fast, simple, good for initial exploration
   * - Expensive evaluations
     - ``BayesianOptimizer``, ``TPEOptimizer``, ``SmacRandomForest``
     - Learn from past evaluations, minimize function calls
   * - Large search space
     - ``RandomSearch``, ``ParticleSwarmOptimizer``
     - Good global coverage without exhaustive search
   * - Multi-modal landscape
     - ``GeneticAlgorithm``, ``DifferentialEvolution``
     - Population-based methods avoid local optima
   * - Small search space
     - ``GridSearch``
     - Exhaustive coverage when feasible
   * - Continuous parameters
     - ``BayesianOptimizer``, ``CmaEsOptimizer``
     - Designed for smooth, continuous spaces
   * - Mixed parameter types
     - ``TPEOptimizer``, ``SmacRandomForest``, ``RandomSearch``
     - Handle categorical + continuous well

----

Configuration
-------------

All optimizers share common parameters and configuration options.

.. seealso::

   :doc:`configuration` covers common parameters (``n_iter``, ``random_state``, ``initialize``),
   warm starting, and performance tips.


.. toctree::
   :maxdepth: 1
   :hidden:

   local_search
   global_search
   population_based
   sequential_model_based
   optuna
   smac
   configuration
