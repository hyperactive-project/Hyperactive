.. _examples_population_based:

===========================
Population-Based Algorithms
===========================

Population-based algorithms maintain multiple candidate solutions simultaneously,
using mechanisms inspired by natural evolution or swarm behavior to explore
the search space efficiently.


Algorithm Examples
------------------

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


When to Use Population-Based Methods
------------------------------------

Population-based algorithms are best suited for:

- **Complex, multimodal landscapes** with many local optima
- **Parallelizable evaluations** where multiple candidates can be evaluated simultaneously
- **Robust optimization** where diversity helps avoid premature convergence
- **Large search spaces** requiring extensive exploration

See :ref:`user_guide_optimizers` for detailed algorithm descriptions.
